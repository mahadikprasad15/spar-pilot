"""Scientific GRPO training engine; implemented separately from workflow orchestration."""
class TrainingDependencies:
    def load_training(self,plan,settings,directory):
        raise NotImplementedError('real GRPO training backend is pending')
