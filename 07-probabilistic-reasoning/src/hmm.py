import numpy as np


class HiddenMarkovModel:

    def __init__(
        self,
        states,
        observations,
        start_prob,
        transition_prob,
        emission_prob
    ):
        self.states = states
        self.observations = observations
        self.start = start_prob
        self.transition = transition_prob
        self.emission = emission_prob

    def forward(self, sequence):

        alpha = {
            s:self.start[s]*self.emission[s][sequence[0]]
            for s in self.states
        }

        for obs in sequence[1:]:
            alpha = {
                s:self.emission[s][obs] *
                sum(
                    alpha[p]*self.transition[p][s]
                    for p in self.states
                )
                for s in self.states
            }

        return sum(alpha.values())
    
    def viterbi(self, sequence):

        path = {
            s:[s]
            for s in self.states
        }

        prob = {
            s:self.start[s]*self.emission[s][sequence[0]]
            for s in self.states
        }

        for obs in sequence[1:]:

            new_prob={}
            new_path={}

            for s in self.states:

                values=[
                    (
                        prob[p]*self.transition[p][s],
                        p
                    )
                    for p in self.states
                ]

                best,prev=max(values)

                new_prob[s]=best*self.emission[s][obs]

                new_path[s]=path[prev]+[s]

            prob=new_prob
            path=new_path

        best=max(prob,key=prob.get)

        return path[best]