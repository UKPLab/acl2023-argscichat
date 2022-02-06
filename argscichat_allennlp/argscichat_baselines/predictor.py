import json

from allennlp.common import JsonDict
from allennlp.predictors import Predictor
from overrides import overrides


@Predictor.register('argscichat')
class ArgSciChatPredictor(Predictor):
    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        to_return = {
            "predicted_answer": outputs["predicted_answers"],
            'article_id': outputs['article_id'],
            'pair_id': outputs['pair_id']
        }

        if 'predicted_evidence' in outputs:
            to_return['predicted_evidence'] = outputs['predicted_evidence']

        return json.dumps(to_return) + "\n"
