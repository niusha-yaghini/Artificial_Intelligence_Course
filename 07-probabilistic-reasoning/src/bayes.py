class BayesCalculator:

    def __init__(
        self,
        prior,
        likelihood,
        evidence
    ):
        self.prior = prior
        self.likelihood = likelihood
        self.evidence = evidence


    def posterior(self):
        return (
            self.likelihood
            *
            self.prior
            /
            self.evidence
        )