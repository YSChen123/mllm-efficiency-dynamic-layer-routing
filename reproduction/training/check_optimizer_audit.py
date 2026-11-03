import types,unittest,torch
from optimizer_audit import optimizer_probe,scheduler_probe

class OptimizerAuditTest(unittest.TestCase):
    def test_probe_handles_zero_cleared_param_groups_without_mutation(self):
        value=torch.nn.Parameter(torch.ones(4));optimizer=torch.optim.AdamW([value])
        value.sum().backward();optimizer.step()
        engine=types.SimpleNamespace(optimizer=types.SimpleNamespace(optimizer=optimizer))
        before=optimizer_probe(engine)
        optimizer.param_groups[0]['params']=[]
        self.assertEqual(optimizer_probe(engine),before)
        self.assertEqual(optimizer.param_groups[0]['params'],[])
    def test_scheduler_may_be_owned_by_trainer_instead_of_deepspeed(self):
        value=torch.nn.Parameter(torch.ones(4));optimizer=torch.optim.AdamW([value])
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
        trainer=types.SimpleNamespace(model_wrapped=types.SimpleNamespace(lr_scheduler=None),lr_scheduler=scheduler)
        self.assertEqual(scheduler_probe(trainer),scheduler.state_dict())

if __name__=='__main__':unittest.main()
