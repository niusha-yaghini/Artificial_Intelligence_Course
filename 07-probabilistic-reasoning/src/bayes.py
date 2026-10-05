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
        

# --------------------------
# Basian Networks
# --------------------------

class Node:
    def __init__(
        self,
        name
    ):
        self.name = name
        self.parents = []
        self.children = []
        self.cpt = {}

    def add_parent(
        self,
        parent
    ):
        self.parents.append(parent)
        parent.children.append(self)
        
def joint_probability(
    cloudy,
    rain,
    wet_grass
):
    probability = 1

    # Cloudy probability
    if cloudy:
        probability *= 0.5
    else:
        probability *= 0.5

    # Rain probability
    if cloudy:
        if rain:
            probability *= 0.8
        else:
            probability *= 0.2
    else:
        if rain:
            probability *= 0.2
        else:
            probability *= 0.8

    # Wet Grass probability
    if rain:
        if wet_grass:
            probability *= 0.9
        else:
            probability *= 0.1
    else:
        if wet_grass:
            probability *= 0.1
        else:
            probability *= 0.9

    return probability


class BayesianNetwork:

    def __init__(
        self,
        data=None
    ):

        self.nodes = {}

        self.data = data


    def add_node(
        self,
        node
    ):

        self.nodes[node.name] = node


    def add_edge(
        self,
        parent,
        child
    ):

        self.nodes[child].parents.append(
            self.nodes[parent]
        )

        self.nodes[parent].children.append(
            self.nodes[child]
        )


    def set_cpt(
        self,
        node_name,
        cpt
    ):

        self.nodes[node_name].cpt = cpt


    def fit(
        self,
        data
    ):

        self.data = data


    def query(
        self,
        variable,
        evidence
    ):

        data = self.data.copy()


        for key,value in evidence.items():

            data = data[
                data[key] == value
            ]

        print(data)

        return data[variable].mean()  
    
    def learn_cpt(self):

        for node_name, node in self.nodes.items():

            if len(node.parents) == 0:

                node.cpt = (
                    self.data[node_name]
                    .value_counts(normalize=True)
                    .to_dict()
                )

            else:

                parent_names = [
                    p.name
                    for p in node.parents
                ]


                cpt = (
                    self.data
                    .groupby(
                        parent_names + [node_name]
                    )
                    .size()
                    .groupby(
                        level=parent_names
                    )
                    .apply(
                        lambda x: x / x.sum()
                    )
                )


                node.cpt = cpt