import math
import unittest
from types import SimpleNamespace
import eval_policy
from eval_policy import expected_layers, validate_training_rows, validate_telemetry, track_requests


class EvaluationPolicyTests(unittest.TestCase):
    def test_summary_write_failure_does_not_stop_following_work(self):
        errors=[]
        def fail():raise OSError('summary volume temporarily unavailable')
        self.assertFalse(eval_policy.best_effort_collect(fail,lambda error:errors.append(str(error))))
        following=[]
        self.assertTrue(eval_policy.best_effort_collect(lambda:following.append('continued'),errors.append))
        self.assertEqual(following,['continued'])
        self.assertEqual(errors,['summary volume temporarily unavailable'])

    def test_dense_uses_stock_cli_and_adaptive_uses_dict_response_cli(self):
        self.assertEqual(eval_policy.evaluation_module('dense'),'lmms_eval')
        self.assertEqual(eval_policy.evaluation_module('reproduced'),'adallava.eval.run_lmms_eval')
        self.assertEqual(eval_policy.evaluation_module('official'),'adallava.eval.run_lmms_eval')

    def test_quantized_endpoints_and_off_grid_budgets(self):
        self.assertEqual([expected_layers(x) for x in [1, .9, .85, .8, .7, .6, .55, .5]],
                         [32, 28, 27, 25, 22, 19, 17, 16])
        for x in [.49, 1.01, float('nan')]:
            with self.assertRaises(ValueError): expected_layers(x)

    def training(self):
        return [{'event':'ENTRY','mode':'formal'},
                {'event':'STEP','step':1}, {'event':'LOG','step':1,'loss':1.2},
                {'event':'STEP','step':2}, {'event':'LOG','step':2,'loss':.8},
                {'event':'TRAIN_LOOP_END','step':2,'max_steps':2},
                {'event':'TRAINING_TOTAL','mode':'formal','wall_seconds':10,'gpu_hours':.01}]

    def test_training_requires_all_steps_and_successful_export_tail(self):
        self.assertTrue(validate_training_rows(self.training(), total_steps=2))
        for bad in [self.training()[:-1], self.training()+[{'event':'FAILED'}],
                    [r for r in self.training() if not (r['event']=='STEP' and r['step']==1)]]:
            with self.assertRaises(ValueError): validate_training_rows(bad,total_steps=2)

    def measurement(self, index=0):
        return {'doc_id':index,'target_budget':.85,'scheduler_observed_budget':.8501,
                'generation_wall_seconds':.2,'generation_cuda_event_ms':190.,
                'lm_prefill_cuda_ms':12.,'vision_plus_lm_prefill_cuda_ms':15.,
                'executed_layers':27,'total_layers':32,'layer_execution_mask':[1]*27+[0]*5}

    def test_telemetry_requires_complete_unique_samples_and_valid_routing(self):
        validate_telemetry([self.measurement(),self.measurement(1)],2,.85,'reproduced')
        bad=self.measurement();bad['layer_execution_mask']=[0]+[1]*27+[0]*4
        for rows in [[self.measurement()], [self.measurement(),self.measurement()], [bad,self.measurement(1)]]:
            with self.assertRaises(ValueError):validate_telemetry(rows,2,.85,'reproduced')

    def test_telemetry_rejects_missing_prefill_or_nan(self):
        for field,value in [('lm_prefill_cuda_ms',0),('generation_wall_seconds',float('nan'))]:
            bad=self.measurement();bad[field]=value
            with self.assertRaises(ValueError):validate_telemetry([bad],1,.85,'reproduced')

    def test_doc_tracking_preserves_visual_call_and_restores_arguments(self):
        seen=[];image=object();doc=object()
        def visual(given):
            self.assertIs(given,doc);return [image]
        req=SimpleNamespace(arguments=('prompt',{},visual,17,'scienceqa_img','test'))
        original=req.arguments
        with track_requests([req],lambda row:seen.append(row)):
            self.assertEqual(req.arguments[2](doc),[image])
        self.assertIs(req.arguments,original)
        self.assertEqual(seen,[{'doc_id':17,'task':'scienceqa_img','split':'test'}])

    def test_doc_tracking_restores_after_generation_failure(self):
        req=SimpleNamespace(arguments=('p',{},lambda doc:doc,0,'mme','test'))
        original=req.arguments
        with self.assertRaises(RuntimeError):
            with track_requests([req],lambda row:None):raise RuntimeError('generation failed')
        self.assertIs(req.arguments,original)


if __name__=='__main__':unittest.main()
