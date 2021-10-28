from typing import Any, Dict, List

import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward
from allennlp.nn import util
from allennlp.training.metrics import Average
from allennlp_models.rc.tools import squad
from overrides import overrides
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np

from argscichat_allennlp.argscichat_baselines.dataset_reader import AnswerType


@Model.register("argscichat_baseline")
class ArgSciChatBaseline(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            transformer_model_name: str,
            attention_dropout: float = 0.1,
            attention_window_size: int = 1024,
            gradient_checkpointing: bool = False,
            evidence_feedforward: FeedForward = None,
            use_evidence_scaffold: bool = True,
            include_argument_mask: bool = True,
            argument_feedforward: FeedForward = None,
            **kwargs
    ):
        super().__init__(vocab, **kwargs)
        config = AutoConfig.from_pretrained(transformer_model_name)
        config.attention_dropout = attention_dropout
        config.attention_window = [attention_window_size] * len(config.attention_window)
        config.gradient_checkpointing = gradient_checkpointing
        self.transformer = AutoModelForSeq2SeqLM.from_pretrained(transformer_model_name, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            transformer_model_name,
            add_special_tokens=False
        )

        if use_evidence_scaffold and include_argument_mask:
            print("Disabling evidence scaffold since argument mask is enabled!")
            use_evidence_scaffold = False

        if evidence_feedforward:
            self.evidence_feedforward = evidence_feedforward
            assert evidence_feedforward.get_output_dim() == 2
        else:
            self.evidence_feedforward = torch.nn.Linear(
                self.transformer.config.hidden_size, 2
            )

        self._use_evidence_scaffold = use_evidence_scaffold

        if argument_feedforward:
            self.argument_feedforward = argument_feedforward
            assert argument_feedforward.get_output_dim() == 2
        else:
            self.argument_feedforward = torch.nn.Linear(
                self.transformer.config.hidden_size, 2
            )

        self._include_argument_mask = include_argument_mask
        self._answer_f1 = Average()
        self._answer_f1_by_type = {answer_type: Average() for answer_type in AnswerType}
        self._evidence_f1 = Average()
        self._evidence_loss = Average()
        self._argument_loss = Average()
        self._argument_f1 = Average()

    def forward(
            self,
            question_with_context: TextFieldTensors,
            paragraph_indices: torch.Tensor,
            global_attention_mask: torch.Tensor = None,
            evidence: torch.Tensor = None,
            argument_mask: torch.Tensor = None,
            answer: TextFieldTensors = None,
            metadata: Dict[str, Any] = None,
    ) -> Dict[str, torch.Tensor]:
        input_ids = util.get_token_ids_from_text_field_tensors(question_with_context)
        attention_mask = util.get_text_field_mask(question_with_context)

        if answer is not None:
            answer_ids = util.get_token_ids_from_text_field_tensors(answer)
        else:
            answer_ids = None

        output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            labels=answer_ids,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        encoded_tokens = output["encoder_last_hidden_state"]

        output_dict = {}
        output_dict["answer_logits"] = output["logits"]
        output_dict['pair_id'] = [item['pair_id'] for item in metadata]
        output_dict['article_id'] = [item['article_id'] for item in metadata]
        loss = None
        if answer is not None:
            loss = output['loss']
            if not self.training:

                generated_token_ids = self.transformer.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    max_length=100
                )
                predicted_answers = [
                    self.tokenizer.decode(generated_token_ids[i].tolist(), skip_special_tokens=True)
                    for i in range(generated_token_ids.size(0))
                ]
                output_dict["predicted_answers"] = predicted_answers
                gold_answers = [instance_metadata["all_answers"] for instance_metadata in metadata]
                for predicted_answer, gold_answer in zip(predicted_answers, gold_answers):
                    f1s_with_types = []
                    for gold_answer_info in gold_answer:
                        f1 = squad.compute_f1(predicted_answer, gold_answer_info['text'])
                        f1s_with_types.append((f1, gold_answer_info['type']))

                    max_f1, max_f1_answer_type = sorted(f1s_with_types, key=lambda x: x[0])[-1]
                    self._answer_f1(max_f1)
                    self._answer_f1_by_type[max_f1_answer_type](max_f1)

        # Facts selection
        if self._use_evidence_scaffold and evidence is not None:
            paragraph_indices = paragraph_indices.squeeze(-1)
            encoded_paragraph_tokens = util.batched_index_select(encoded_tokens.contiguous(), paragraph_indices)
            evidence_logits = self.evidence_feedforward(encoded_paragraph_tokens)
            evidence_mask = paragraph_indices != -1

            # Use a loss function that gives higher weight to the positive classes
            weights = torch.tensor(
                [
                    evidence.sum() + 1,
                    evidence_mask.sum() - evidence.sum() + 1,
                ],
                device=evidence_logits.device,
                dtype=evidence_logits.dtype,
            )
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
            evidence_loss = loss_fn(evidence_logits.view(-1, 2), evidence.view(-1))
            self._evidence_loss(float(evidence_loss.detach().cpu()))
            if loss is None:
                loss = evidence_loss
            else:
                loss = loss + evidence_loss
            if not self.training:
                predicted_evidence_indices = evidence_logits.argmax(dim=-1).tolist()
                gold_evidence_indices = [instance_metadata["all_evidence_masks"]
                                         for instance_metadata in metadata]
                for evidence_f1 in self._compute_evidence_f1(predicted_evidence_indices,
                                                             gold_evidence_indices):
                    self._evidence_f1(evidence_f1)
                output_dict['predicted_evidence'] = predicted_evidence_indices

        # Argument selection
        if self._include_argument_mask and argument_mask is not None and not self._use_evidence_scaffold:
            paragraph_indices = paragraph_indices.squeeze(-1)
            encoded_paragraph_tokens = util.batched_index_select(encoded_tokens.contiguous(), paragraph_indices)

            argument_logits = self.argument_feedforward(encoded_paragraph_tokens)

            argument_weight_mask = paragraph_indices != -1

            # Use a loss function that gives higher weight to the positive classes
            weights = torch.tensor(
                [
                    argument_mask.sum() + 1,
                    argument_weight_mask.sum() - argument_mask.sum() + 1,
                ],
                device=argument_logits.device,
                dtype=argument_logits.dtype,
            )

            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
            argument_loss = loss_fn(argument_logits.view(-1, 2), argument_mask.view(-1))

            self._argument_loss(float(argument_loss.detach().cpu()))
            if loss is None:
                loss = argument_loss
            else:
                loss = loss + argument_loss

            if not self.training:
                predicted_argument_indices = argument_logits.argmax(dim=-1).tolist()
                for argument_f1 in self._compute_evidence_f1(predicted_argument_indices,
                                                             [argument_mask]):
                    self._argument_f1(argument_f1)
                output_dict['predicted_arguments'] = predicted_argument_indices

                # Compute cosine similarity between query and candidates
                query_index = torch.zeros_like(paragraph_indices)[:, 0].view(-1, 1)
                encoded_query_token = util.batched_index_select(encoded_tokens.contiguous(), query_index)
                norm_encoded_query_token = encoded_query_token / encoded_query_token.norm(dim=-1)[:, :, None]
                norm_encoded_paragraph_tokens = encoded_paragraph_tokens / encoded_paragraph_tokens.norm(dim=-1)[:, :,
                                                                           None]

                cosine_similarity = torch.matmul(norm_encoded_query_token,
                                                 norm_encoded_paragraph_tokens.transpose(1, 2))
                cosine_similarity = cosine_similarity.squeeze(1).tolist()

                gold_evidence_indices = [instance_metadata["all_evidence_masks"]
                                         for instance_metadata in metadata]
                evidence_f1, evidence_indices = self._compute_arg_evidence_f1(cosine_similarity,
                                                                             predicted_argument_indices,
                                                                             gold_evidence_indices)

                for value in evidence_f1:
                    self._evidence_f1(value)

                output_dict['predicted_evidence'] = evidence_indices

        output_dict["loss"] = loss
        return output_dict

    @staticmethod
    def _compute_evidence_f1(
            predicted_evidence_indices: List[List[int]],
            gold_evidence_indices: List[List[List[int]]]
    ) -> List[float]:
        f1s = []
        for instance_predicted, instance_gold in zip(predicted_evidence_indices, gold_evidence_indices):
            instance_f1s = []
            for gold in instance_gold:
                # Skip example without facts
                if sum(gold) == 0:
                    break
                # If the document was truncated to fit in the model, the gold will be longer than the
                # predicted indices.
                predicted = instance_predicted + [0] * (len(gold) - len(instance_predicted))
                true_positives = sum([i and j for i, j in zip(predicted, gold)])
                if sum(predicted) == 0:
                    precision = 1.0 if sum(gold) == 0 else 0.0
                else:
                    precision = true_positives / sum(predicted)
                recall = true_positives / sum(gold) if sum(gold) != 0 else 1.0
                if precision + recall == 0:
                    instance_f1s.append(0.0)
                else:
                    instance_f1s.append(2 * precision * recall / (precision + recall))
            if len(instance_f1s):
                f1s.append(max(instance_f1s))
        return f1s

    @staticmethod
    def _compute_arg_evidence_f1(
            similarities: List[List[float]],
            predicted_argument_indices: List[List[int]],
            gold_evidence_indices: List[List[List[int]]]
    ) -> (List[float], List[int]):
        f1s = []
        predicted_indices = []
        for instance_similarities, instance_argument_indices, instance_gold in zip(similarities,
                                                                                   predicted_argument_indices,
                                                                                   gold_evidence_indices):

            assert len(instance_similarities) == len(instance_argument_indices)

            instance_argument_indices = np.where(instance_argument_indices)[0]
            sorted_sim_indices = np.argsort(instance_similarities, axis=0)

            instance_f1s = []
            for gold in instance_gold:
                total_facts = sum(gold)
                if total_facts == 0:
                    break

                assert len(gold) == len(instance_similarities), "Gold: {0}, Similarities: {1}".format(len(gold), len(instance_similarities))

                best_indexes = [idx for idx in sorted_sim_indices if idx in instance_argument_indices]
                best_indexes = best_indexes[-total_facts:]

                if not len(best_indexes):
                    instance_f1s.append(0.0)
                else:
                    predicted = [1 if idx in best_indexes else 0 for idx in range(len(gold))]
                    true_positives = sum([i and j for i, j in zip(predicted, gold)])
                    if sum(predicted) == 0:
                        precision = 1.0 if sum(gold) == 0 else 0.0
                    else:
                        precision = true_positives / sum(predicted)
                    recall = true_positives / sum(gold) if sum(gold) != 0 else 1.0
                    if precision + recall == 0:
                        instance_f1s.append(0.0)
                    else:
                        instance_f1s.append(2 * precision * recall / (precision + recall))

            if len(instance_f1s):
                predicted_indices.append(predicted)  # it works since we have just one gold label
                f1s.append(max(instance_f1s))
        return f1s, predicted_indices

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1_score = self._answer_f1.get_metric(reset)
        extractive_f1_score = self._answer_f1_by_type[AnswerType.EXTRACTIVE].get_metric(reset)
        abstractive_f1_score = self._answer_f1_by_type[AnswerType.ABSTRACTIVE].get_metric(reset)
        boolean_f1_score = self._answer_f1_by_type[AnswerType.BOOLEAN].get_metric(reset)
        none_f1_score = self._answer_f1_by_type[AnswerType.NONE].get_metric(reset)
        evidence_f1 = self._evidence_f1.get_metric(reset)
        evidence_loss = self._evidence_loss.get_metric(reset)
        argument_f1 = self._argument_f1.get_metric(reset)
        argument_loss = self._argument_loss.get_metric(reset)
        return {
            "answer_f1": f1_score,
            "extr_f1": extractive_f1_score,
            "abstr_f1": abstractive_f1_score,
            "bool_f1": boolean_f1_score,
            "none_f1": none_f1_score,
            "evidence_f1": evidence_f1,
            "evidence_loss": evidence_loss,
            'argument_f1': argument_f1,
            'argument_loss': argument_loss
        }
