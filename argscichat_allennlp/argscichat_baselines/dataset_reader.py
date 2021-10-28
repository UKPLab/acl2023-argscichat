import json
import logging
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Iterable, Tuple

import torch
from allennlp.common.file_utils import cached_path, open_compressed
from allennlp.common.util import JsonDict
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    MetadataField,
    TextField,
    IndexField,
    ListField,
    TensorField,
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer
from overrides import overrides

logger = logging.getLogger(__name__)


class AnswerType(Enum):
    EXTRACTIVE = 1
    ABSTRACTIVE = 2
    BOOLEAN = 3
    NONE = 4


@DatasetReader.register('argscichat')
class ArgSciChatReader(DatasetReader):

    def __init__(
            self,
            transformer_model_name: str = "allenai/led-base-16384",
            max_query_length: int = 500,
            max_document_length: int = 3300,
            paragraph_separator: Optional[str] = "</s>",
            include_global_attention_mask: bool = True,
            include_argument_mask: bool = False,
            argument_mask_threshold: str = None,
            context: list = ["query"],
            for_training: bool = False,
            **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )
        self._transformer_model_name = transformer_model_name
        self._tokenizer = PretrainedTransformerTokenizer(
            transformer_model_name, add_special_tokens=False
        )

        self._include_global_attention_mask = include_global_attention_mask
        self._include_argument_mask = include_argument_mask
        self._argument_mask_threshold = argument_mask_threshold
        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(transformer_model_name)
        }
        self.max_query_length = max_query_length
        self.max_document_length = max_document_length
        self._paragraph_separator = paragraph_separator

        assert 'query' in context

        for item in context:
            assert item.lower() in ['query', 'article', 'history', 'facts'], "Found {0}".format(item)

        self._context = [item.lower() for item in context]
        self._for_training = for_training
        self._stats = defaultdict(int)

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading the dataset")
        if file_path.endswith(".json"):
            yield from self._read_json(file_path)
        elif file_path.endswith(".jsonl"):
            yield from self._read_json_lines(file_path)
        else:
            raise RuntimeError(
                f"Unsupported extension on file: {file_path}. Only json and jsonl are supported."
            )

    def _read_json(self, file_path: str):
        logger.info("Reading json file at %s", file_path)
        with open_compressed(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        for article_id, article in self.shard_iterable(dataset.items()):
            if not article["content"]:
                continue
            article["article_id"] = article_id
            yield from self._article_to_instances(article)
        self._log_stats()

    def _read_json_lines(self, file_path: str):
        logger.info("Reading json lines file at %s", file_path)
        with open_compressed(file_path) as dataset_file:
            for data_line in self.shard_iterable(dataset_file):
                data = json.loads(data_line)
                yield from self._article_to_instances(data)
        self._log_stats()

    def _log_stats(self) -> None:
        logger.info("Stats:")
        for key, value in self._stats.items():
            logger.info("%s: %d", key, value)

    def _article_to_instances(self, article: Dict[str, Any]) -> Iterable[Instance]:
        paragraphs = self._get_paragraphs_from_article(article)

        if 'article' in self._context:
            assert len(paragraphs) == len(article['content'])
            # print('Paragraphs: {0}, Content: {1}'.format(len(paragraphs), len(article['content'])))

        tokenized_context = None
        paragraph_start_indices = None

        if self._include_argument_mask:
            argument_mask = article['argument_mask_{}'.format(self._argument_mask_threshold)]
            # print('Argument mask: {0}, Expected: {1}'.format(len(argument_mask), len(article['argument_mask'])))
            assert len(argument_mask) == len(paragraphs), 'Mask = {0} -- Paragraphs = {1}, -- Article {2}'.format(
                len(argument_mask), len(paragraphs), len(article['content']))
        else:
            argument_mask = None

        self._stats["number of documents"] += 1
        for pair in article["message_pairs"]:
            # print('*' * 20)
            self._stats["number of messages"] += 1

            facts, history, reply = self._extract_pair_data(pair)
            facts = self._validate_facts(facts=facts, paragraphs=paragraphs, title=article['id'])
            # print('Paragraphs: {0}, Content: {1}'.format(len(paragraphs), len(article['content'])))

            if 'history' in self._context:
                pair_paragraphs = history + paragraphs
                # print('Paragraphs: {0}, Content: {1}, History: {2}'.format(len(pair_paragraphs), len(article['content']), len(history)))
                assert len(pair_paragraphs) == len(article['content']) + len(history)

                if self._include_argument_mask:
                    # print('Argument mask: {0}, Expected: {1}'.format(len(argument_mask), len(article['argument_mask'])))
                    pair_argument_mask = [0] * len(history) + argument_mask
                    assert len(pair_argument_mask) == len(pair_paragraphs), 'Mask = {0} -- Paragraphs = {1}'.format(
                        len(pair_argument_mask), len(pair_paragraphs))
                else:
                    pair_argument_mask = argument_mask
            else:
                pair_paragraphs = paragraphs
                pair_argument_mask = argument_mask

            if len(pair_paragraphs):
                tokenized_context, paragraph_start_indices = self._tokenize_paragraphs(
                    pair_paragraphs
                )

                # Ignore history!
                if 'history' in self._context:
                    assert len(paragraph_start_indices) == len(pair_paragraphs), "Expected {0}, Paragraphs {1}".format(len(paragraph_start_indices), len(pair_paragraphs))
                    paragraph_start_indices = paragraph_start_indices[len(history):]
                    assert len(paragraph_start_indices) == len(article['content']), "Expected {0}, Got {1}".format(len(article['content']), len(paragraph_start_indices))

                    if self._include_argument_mask:
                        pair_argument_mask = pair_argument_mask[len(history):]
                        assert len(paragraph_start_indices) == len(pair_argument_mask)

            facts_mask = self._get_facts_mask(facts, pair_paragraphs, has_article='article' in self._context)

            if 'history' in self._context:
                facts_mask = facts_mask[len(history):]

            additional_metadata = {
                "pair_id": pair['id'],
                "article_id": article.get("article_id"),
                "all_answers": [{'text': reply, 'type': AnswerType.ABSTRACTIVE}],
                "all_evidence_masks": [facts_mask]
            }

            if paragraph_start_indices is not None:
                assert len(facts_mask) == len(paragraph_start_indices), "Facts: {0}, Paragraphs indices: {1}".format(len(facts_mask), len(paragraph_start_indices))
            if pair_argument_mask is not None:
                assert len(facts_mask) == len(pair_argument_mask)

            yield self.text_to_instance(
                pair["P_Message"],
                pair_paragraphs,
                tokenized_context,
                paragraph_start_indices,
                facts_mask,
                reply,
                facts,
                pair_argument_mask,
                additional_metadata
            )

    def _get_paragraphs_from_article(self, article: JsonDict) -> List[str]:
        paragraphs = []
        if 'article' in self._context:
            for sentence in article['content']:
                paragraphs.append(sentence.replace("\n", " ").strip())

        return paragraphs

    def _tokenize_paragraphs(
            self, paragraphs: List[str]
    ) -> Tuple[List[Token], List[int]]:
        tokenized_context = []
        paragraph_start_indices = []
        for paragraph in paragraphs:
            tokenized_paragraph = self._tokenizer.tokenize(paragraph)
            paragraph_start_indices.append(len(tokenized_context))
            tokenized_context.extend(tokenized_paragraph)
            if self._paragraph_separator:
                tokenized_context.append(Token(self._paragraph_separator))
        if self._paragraph_separator:
            # We added the separator after every paragraph, so we remove it after the last one.
            tokenized_context = tokenized_context[:-1]

        assert len(paragraph_start_indices) == len(paragraphs)
        return tokenized_context, paragraph_start_indices

    def _validate_facts(self, facts, paragraphs, title):

        if not len(paragraphs) or 'article' not in self._context:
            original_length = len(facts)
            facts = [fact for fact in facts if fact not in title]
            if original_length != len(facts):
                print('Some facts were discarded due to being in the title section!'
                      ' Original: {0} -- New: {1}'.format(original_length, len(facts)))

            return facts

        missing_indexes = []
        for fact_idx, fact in enumerate(facts):
            found = False
            for paragraph in paragraphs:
                if fact.lower() in paragraph.lower() or paragraph.lower() in fact.lower():
                    found = True
                    break

            if not found:
                missing_indexes.append(fact_idx)

        if len(missing_indexes) > 0:
            print('Found potential issue concerning facts retrieval: Got {0} -- Expected {1}.'
                  ' Verifying if missing fact(s) are in title...'.format(len(facts) - len(missing_indexes), len(facts)))
            to_avoid = []
            for fact_idx in missing_indexes:
                if facts[fact_idx] in title:
                    to_avoid.append(fact_idx)

            if len(to_avoid) == len(missing_indexes):
                print('Inconsistency has been debunked!')
            else:
                missing_facts = [facts[idx] for idx in missing_indexes if idx not in to_avoid]
                print('Inconsistency remains ({0} candidates): \n{1}'.format(len(missing_indexes) - len(to_avoid),
                                                                             missing_facts))

        return [facts[idx] for idx in range(len(facts)) if idx not in missing_indexes]

    def _extract_pair_data(
            self, pair: JsonDict
    ) -> Tuple[List[str], List[str], str]:
        facts_spans = [x.replace("\n", " ").strip() for x in pair["facts"]]
        facts_spans = [x for x in facts_spans if x != ""]
        if not facts_spans:
            self._stats["replies_with_no_facts"] += 1
        if len(facts_spans) > 1:
            self._stats["multiple_facts_count"] += 1

        history = [x.replace("\n", " ").strip() for x in pair["history"]]
        history = [x for x in history if x != ""]

        return facts_spans, history, pair['DE_Message']

    @staticmethod
    def _get_facts_mask(facts: List[str], paragraphs: List[str], has_article) -> List[int]:
        """
        Takes a list of evidence snippets, and the list of all the paragraphs from the
        paper, and returns a list of indices of the paragraphs that contain the evidence.
        """

        if not len(paragraphs) or not has_article:
            return [0] * len(paragraphs)

        facts_mask = [0] * len(paragraphs)
        total_found = 0
        for fact in facts:
            for paragraph_idx, paragraph in enumerate(paragraphs):
                if fact.lower() in paragraph.lower() or paragraph.lower() in fact.lower():
                    facts_mask[paragraph_idx] = 1
                    total_found += 1
                    break

        assert total_found == len(facts), "Got {0} facts but facts mask has {1} 1s".format(len(facts), total_found)
        return facts_mask

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["question_with_context"].token_indexers = self._token_indexers
        instance.fields["answer"].token_indexers = self._token_indexers

    @overrides
    def text_to_instance(
            self,  # type: ignore  # pylint: disable=arguments-differ
            P_Message: str,
            paragraphs: List[str],
            tokenized_context: List[Token] = None,
            paragraph_start_indices: List[int] = None,
            facts_mask: List[int] = None,
            reply: str = None,
            facts: List[str] = None,
            argument_mask: List[int] = None,
            additional_metadata: Dict[str, Any] = None,
    ) -> Instance:
        fields = {}

        tokenized_question = self._tokenizer.tokenize(P_Message)
        if len(tokenized_question) > self.max_query_length:
            self._stats["number of truncated questions"] += 1
            tokenized_question = tokenized_question[:self.max_query_length]

        if tokenized_context is None or paragraph_start_indices is None:
            if 'facts' in self._context:
                tokenized_context, paragraph_start_indices = self._tokenize_paragraphs(
                    facts
                )
            else:
                tokenized_context, paragraph_start_indices = self._tokenize_paragraphs(
                    paragraphs
                )

        allowed_context_length = (
                self.max_document_length
                - len(tokenized_question)
                - len(self._tokenizer.sequence_pair_start_tokens)
                - 1  # for paragraph separator
        )
        if len(tokenized_context) > allowed_context_length:
            self._stats["number of truncated contexts"] += 1
            tokenized_context = tokenized_context[:allowed_context_length]
            paragraph_start_indices = [index for index in paragraph_start_indices
                                       if index <= allowed_context_length]
            num_paragraphs = len(paragraph_start_indices)
            if facts_mask is not None:
                facts_mask = facts_mask[:num_paragraphs]

            if argument_mask is not None:
                argument_mask = argument_mask[:num_paragraphs]

        # This is what Iz's code does.
        question_and_context = (
                self._tokenizer.sequence_pair_start_tokens
                + tokenized_question
                + [Token(self._paragraph_separator)]
                + tokenized_context
        )
        # make the question field
        question_field = TextField(question_and_context)
        fields["question_with_context"] = question_field

        start_of_context = (
                len(self._tokenizer.sequence_pair_start_tokens)
                + len(tokenized_question)
        )

        paragraph_indices_list = [x + start_of_context for x in paragraph_start_indices]

        paragraph_indices_field = ListField(
            [IndexField(x, question_field) for x in paragraph_indices_list] if paragraph_indices_list else
            [IndexField(-1, question_field)]
        )

        fields["paragraph_indices"] = paragraph_indices_field

        if self._include_global_attention_mask:
            # We need to make a global attention array. We'll use all the paragraph indices and the
            # indices of question tokens.
            mask_indices = set(list(range(start_of_context)) + paragraph_indices_list)
            mask = [
                True if i in mask_indices else False for i in range(len(question_field))
            ]
            fields["global_attention_mask"] = TensorField(torch.tensor(mask))

        if facts_mask is not None:
            evidence_field = TensorField(torch.tensor(facts_mask))
            fields["evidence"] = evidence_field

        if argument_mask is not None:
            argument_mask_field = TensorField(torch.tensor(argument_mask))
            fields['argument_mask'] = argument_mask_field

        if reply:
            fields["answer"] = TextField(
                self._tokenizer.add_special_tokens(self._tokenizer.tokenize(reply))
            )

        # make the metadata
        metadata = {
            "question": P_Message,
            "question_tokens": tokenized_question,
            "paragraphs": paragraphs,
            "context_tokens": tokenized_context,
        }
        if additional_metadata is not None:
            metadata.update(additional_metadata)
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)


@DatasetReader.register('sentence_argscichat')
class SentenceArgSciChatReader(ArgSciChatReader):

    def _article_to_instances(self, article: Dict[str, Any]) -> Iterable[Instance]:
        paragraphs = self._get_paragraphs_from_article(article)

        if 'article' in self._context:
            assert len(paragraphs) == len(article['content'])
            # print('Paragraphs: {0}, Content: {1}'.format(len(paragraphs), len(article['content'])))

        tokenized_context = None
        paragraph_start_indices = None

        if self._include_argument_mask:
            argument_mask = article['argument_mask_{}'.format(self._argument_mask_threshold)]
            # print('Argument mask: {0}, Expected: {1}'.format(len(argument_mask), len(article['argument_mask'])))
            assert len(argument_mask) == len(paragraphs), 'Mask = {0} -- Paragraphs = {1}, -- Article {2}'.format(
                len(argument_mask), len(paragraphs), len(article['content']))
        else:
            argument_mask = None

        self._stats["number of documents"] += 1
        for pair in article["message_pairs"]:
            self._stats["number of messages"] += 1

            facts, history, reply, past_replies = self._extract_pair_data(pair)
            facts = self._validate_facts(facts=facts, paragraphs=paragraphs, title=article['id'])

            if 'history' in self._context:
                pair_paragraphs = history + paragraphs
                # print('Paragraphs: {0}, Content: {1}, History: {2}'.format(len(pair_paragraphs), len(article['content']), len(history)))
                assert len(pair_paragraphs) == len(article['content']) + len(history)

                if self._include_argument_mask:
                    # print('Argument mask: {0}, Expected: {1}'.format(len(argument_mask), len(article['argument_mask'])))
                    pair_argument_mask = [0] * len(history) + argument_mask
                    assert len(pair_argument_mask) == len(pair_paragraphs), 'Mask = {0} -- Paragraphs = {1}'.format(
                        len(pair_argument_mask), len(pair_paragraphs))
                else:
                    pair_argument_mask = argument_mask
            else:
                pair_paragraphs = paragraphs
                pair_argument_mask = argument_mask

            if len(pair_paragraphs):
                tokenized_context, paragraph_start_indices = self._tokenize_paragraphs(
                    pair_paragraphs
                )
                # Ignore history!
                if 'history' in self._context:
                    assert len(paragraph_start_indices) == len(pair_paragraphs), "Expected {0}, Paragraphs {1}".format(len(paragraph_start_indices), len(pair_paragraphs))
                    paragraph_start_indices = paragraph_start_indices[len(history):]
                    assert len(paragraph_start_indices) == len(article['content']), "Expected {0}, Got {1}".format(len(article['content']), len(paragraph_start_indices))

                    if self._include_argument_mask:
                        pair_argument_mask = pair_argument_mask[len(history):]
                        assert len(paragraph_start_indices) == len(pair_argument_mask)

            facts_mask = self._get_facts_mask(facts, pair_paragraphs, has_article='article' in self._context)

            if 'history' in self._context:
                facts_mask = facts_mask[len(history):]

            additional_metadata = {
                "pair_id": pair['id'],
                "article_id": article.get("article_id"),
                "all_answers": [{'text': reply, 'type': AnswerType.ABSTRACTIVE}],
                "all_evidence_masks": [facts_mask]
            }

            if paragraph_start_indices is not None:
                assert len(facts_mask) == len(paragraph_start_indices), "Facts: {0}, Paragraphs indices: {1}".format(len(facts_mask), len(paragraph_start_indices))
            if pair_argument_mask is not None:
                assert len(facts_mask) == len(pair_argument_mask)

            yield self.text_to_instance(
                pair["P_Message"],
                past_replies,
                paragraphs,
                tokenized_context,
                paragraph_start_indices,
                facts_mask,
                reply,
                facts,
                argument_mask,
                additional_metadata
            )

    def _extract_pair_data(
            self, pair: JsonDict
    ) -> Tuple[List[str], List[str], str, List[str]]:
        facts_spans = [x.replace("\n", " ").strip() for x in pair["facts"]]
        facts_spans = [x for x in facts_spans if x != ""]
        if not facts_spans:
            self._stats["replies_with_no_facts"] += 1
        if len(facts_spans) > 1:
            self._stats["multiple_facts_count"] += 1

        history = [x.replace("\n", " ").strip() for x in pair["history"]]
        history = [x for x in history if x != ""]

        past_replies = pair['Previous DE sentences']

        return facts_spans, history, pair['DE_Message'], past_replies

    @overrides
    def text_to_instance(
            self,  # type: ignore  # pylint: disable=arguments-differ
            P_Message: str,
            past_replies: List[str],
            paragraphs: List[str],
            tokenized_context: List[Token] = None,
            paragraph_start_indices: List[int] = None,
            facts_mask: List[int] = None,
            reply: str = None,
            facts: List[str] = None,
            argument_mask: List[int] = None,
            additional_metadata: Dict[str, Any] = None,
    ) -> Instance:
        fields = {}

        tokenized_question = self._tokenizer.tokenize(P_Message)

        # Add past replies
        tokenized_replies = []
        for reply in past_replies:
            tokenized_replies.extend(self._tokenizer.tokenize(reply))
            tokenized_replies.append(Token(self._paragraph_separator))

        if len(tokenized_replies):
            tokenized_question += tokenized_replies

        if len(tokenized_question) > self.max_query_length:
            self._stats["number of truncated questions"] += 1
            tokenized_question = tokenized_question[:self.max_query_length]

        if tokenized_context is None or paragraph_start_indices is None:
            if 'facts' in self._context:
                tokenized_context, paragraph_start_indices = self._tokenize_paragraphs(
                    facts
                )
            else:
                tokenized_context, paragraph_start_indices = self._tokenize_paragraphs(
                    paragraphs
                )

        allowed_context_length = (
                self.max_document_length
                - len(tokenized_question)
                - len(self._tokenizer.sequence_pair_start_tokens)
                - 1  # for paragraph separator
        )
        if len(tokenized_context) > allowed_context_length:
            self._stats["number of truncated contexts"] += 1
            tokenized_context = tokenized_context[:allowed_context_length]
            paragraph_start_indices = [index for index in paragraph_start_indices
                                       if index <= allowed_context_length]
            num_paragraphs = len(paragraph_start_indices)

            if facts_mask is not None:
                facts_mask = facts_mask[:num_paragraphs]

            if argument_mask is not None:
                argument_mask = argument_mask[:num_paragraphs]

        # This is what Iz's code does.
        question_and_context = (
                self._tokenizer.sequence_pair_start_tokens
                + tokenized_question
                + [Token(self._paragraph_separator)]
                + tokenized_context
        )
        # make the question field
        question_field = TextField(question_and_context)
        fields["question_with_context"] = question_field

        start_of_context = (
                len(self._tokenizer.sequence_pair_start_tokens)
                + len(tokenized_question)
        )

        paragraph_indices_list = [x + start_of_context for x in paragraph_start_indices]

        paragraph_indices_field = ListField(
            [IndexField(x, question_field) for x in paragraph_indices_list] if paragraph_indices_list else
            [IndexField(-1, question_field)]
        )

        fields["paragraph_indices"] = paragraph_indices_field

        if self._include_global_attention_mask:
            # We need to make a global attention array. We'll use all the paragraph indices and the
            # indices of question tokens.
            mask_indices = set(list(range(start_of_context)) + paragraph_indices_list)
            mask = [
                True if i in mask_indices else False for i in range(len(question_field))
            ]
            fields["global_attention_mask"] = TensorField(torch.tensor(mask))

        if facts_mask is not None:
            evidence_field = TensorField(torch.tensor(facts_mask))
            fields["evidence"] = evidence_field

        if argument_mask is not None:
            argument_mask_field = TensorField(torch.tensor(argument_mask))
            fields['argument_mask'] = argument_mask_field

        if reply:
            fields["answer"] = TextField(
                self._tokenizer.add_special_tokens(self._tokenizer.tokenize(reply))
            )

        # make the metadata
        metadata = {
            "question": P_Message,
            "question_tokens": tokenized_question,
            "paragraphs": paragraphs,
            "context_tokens": tokenized_context,
        }
        if additional_metadata is not None:
            metadata.update(additional_metadata)
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)
