import torch
import json
from scipy.stats import norm

# Function to compute p-value of EAT
def compute_p_value(associations1: torch.tensor, 
                    associations2: torch.tensor, 
                    permutations:int=1000
                    ) -> float:
    """
    Computes the one-tailed p-value of an EAT for the given sets of associations with two attribute groups.
    """

    # Compute test statistic, difference between sums of associations
    test_statistic = associations1.sum() - associations2.sum()
    
    # Create joint distribution of associations by concatenating
    joint_sims = torch.cat([associations1, associations2])

    # Permute joint distribution to form tensor of shape (permutations, len(joint_sims))
    joint_permutations = torch.stack([joint_sims[torch.randperm(len(joint_sims))] for _ in range(permutations)])

    # Compute differential associations for each permutation, choosing first len(associations1) for A and last len(associations2) for B
    differential_associations = joint_permutations[:, :len(associations1)].sum(dim=1) - joint_permutations[:, len(associations1):].sum(dim=1)

    # Compute mean and standard deviation of distribution of permutations
    dist_mean, dist_std = differential_associations.mean(), differential_associations.std(correction=1)

    # Compute p-value as probability of observing a test statistic as extreme as the one observed, given the null hypothesis
    p_value = min(norm.cdf(test_statistic, loc=dist_mean, scale=dist_std), 1 - norm.cdf(test_statistic, loc=dist_mean, scale=dist_std))

    return p_value

# Multilevel EAT
def ML_EAT(A: torch.tensor, 
           B: torch.tensor, 
           X: torch.tensor, 
           Y: torch.tensor, 
           permutations: int=1000,
           eat_name: str='', 
           return_type: str='dict',
           ) -> dict:
    """
    Computes the three levels of the multilevel WEAT test for the given sets of attribute (A, B) and target (X, Y) words.
    """

    # Normalize all vectors to unit length by dividing by the norm
    A_normed = A / A.norm(dim=1, keepdim=True)
    B_normed = B / B.norm(dim=1, keepdim=True)
    X_normed = X / X.norm(dim=1, keepdim=True)
    Y_normed = Y / Y.norm(dim=1, keepdim=True)

    # Compute dot product (cosine similarity, because vectors are normalized) of each target word with each attribute word
    X_A = torch.mm(X_normed, A_normed.t())
    X_B = torch.mm(X_normed, B_normed.t())
    Y_A = torch.mm(Y_normed, A_normed.t())
    Y_B = torch.mm(Y_normed, B_normed.t())

    # Compute mean and standard deviation of each matrix of similarities to characterize the distribution
    X_A_mean, X_A_std = X_A.mean(), X_A.std(correction=1)
    X_B_mean, X_B_std = X_B.mean(), X_B.std(correction=1)
    Y_A_mean, Y_A_std = Y_A.mean(), Y_A.std(correction=1)
    Y_B_mean, Y_B_std = Y_B.mean(), Y_B.std(correction=1)

    # Compute means, corresponding to the mean similarity of each attribute word with all target words in X
    st_means_AX = X_A.mean(dim=0)
    st_means_BX = X_B.mean(dim=0)
    st_means_AY = Y_A.mean(dim=0)
    st_means_BY = Y_B.mean(dim=0)
    
    # Compute intermediate effect size and p-value for X
    joint_X_means = torch.cat([st_means_AX, st_means_BX])
    level2_effect_X = (st_means_AX.mean() - st_means_BX.mean()) / joint_X_means.std(correction=1)
    level2_p_value_X = compute_p_value(st_means_AX, st_means_BX, permutations)

    # Compute intermediate effect size and p-value for Y
    joint_Y_means = torch.cat([st_means_AY, st_means_BY])
    level2_effect_Y = (st_means_AY.mean() - st_means_BY.mean()) / joint_Y_means.std(correction=1)
    level2_p_value_Y = compute_p_value(st_means_AY, st_means_BY, permutations)

    # Compute means, corresponding to the mean similarity of each target word with all attribute words in A and B
    X_means_A = X_A.mean(dim=1)
    X_means_B = X_B.mean(dim=1)
    Y_means_A = Y_A.mean(dim=1)
    Y_means_B = Y_B.mean(dim=1)

    # Compute associations for each target word, corresponding to difference in means of similarities with A and B
    X_diffs = X_means_A - X_means_B
    Y_diffs = Y_means_A - Y_means_B

    # Form joint distribution of associations
    joint_diffs = torch.cat([X_diffs, Y_diffs])

    # Compute effect size - difference in means of associations divided by standard deviation of joint distribution
    X_mean = X_diffs.mean()
    Y_mean = Y_diffs.mean()

    # Note that we use the unbiased estimator of standard deviation, which divides by N-1 instead of N (correction=1)
    joint_std = joint_diffs.std(correction=1)
    level1_effect_size = (X_mean - Y_mean) / joint_std

    # Compute one-tailed p-value by permuting the target word associations and obtaining the difference in means for each permutation
    level1_p_value = compute_p_value(X_diffs, Y_diffs, permutations)

    # Return dictionary of results
    ml_eat_dict = {
        'L3 A-X Mean': X_A_mean.item(),
        'L3 A-X Std': X_A_std.item(),
        'L3 B-X Mean': X_B_mean.item(),
        'L3 B-X Std': X_B_std.item(),
        'L3 A-Y Mean': Y_A_mean.item(),
        'L3 A-Y Std': Y_A_std.item(),
        'L3 B-Y Mean': Y_B_mean.item(),
        'L3 B-Y Std': Y_B_std.item(),
        'L2 Effect Size X': level2_effect_X.item(),
        'L2 p-value X': level2_p_value_X,
        'L2 Effect Size Y': level2_effect_Y.item(),
        'L2 p-value Y': level2_p_value_Y,
        'L1 Effect Size': level1_effect_size.item(),
        'L1 p-value': level1_p_value
    }

    # Return results in specified format
    if return_type == 'string':
        ml_eat_string = f'{eat_name} ML-EAT Results\n\n'
        for key, value in ml_eat_dict.items():
            ml_eat_string += f'{key}: {value}\n'
        return ml_eat_string
    
    elif return_type == 'json':
        return json.dumps(ml_eat_dict)
    
    else:
        return ml_eat_dict

# Standard EAT
def EAT(A: torch.tensor, 
         B: torch.tensor, 
         X: torch.tensor, 
         Y: torch.tensor, 
         permutations: int=1000,
         ) -> tuple[float, float]:
    """
    Computes the effect size and one-tailed p-value of the EAT for the given sets of attribute (A, B) and target (X, Y) words.
    """

    # Normalize all vectors to unit length by dividing by the norm
    A_normed = A / A.norm(dim=1, keepdim=True)
    B_normed = B / B.norm(dim=1, keepdim=True)
    X_normed = X / X.norm(dim=1, keepdim=True)
    Y_normed = Y / Y.norm(dim=1, keepdim=True)

    # Compute dot product (cosine similarity, because vectors are normalized) of each target word with each attribute word
    X_A = torch.mm(X_normed, A_normed.t())
    X_B = torch.mm(X_normed, B_normed.t())
    Y_A = torch.mm(Y_normed, A_normed.t())
    Y_B = torch.mm(Y_normed, B_normed.t())

    # Compute means corresponding to the mean similarity of each target word with all attribute words in A and B
    X_means_A = X_A.mean(dim=1)
    X_means_B = X_B.mean(dim=1)
    Y_means_A = Y_A.mean(dim=1)
    Y_means_B = Y_B.mean(dim=1)

    # Compute associations for each target word, corresponding to difference in means of similarities with A and B
    X_diffs = X_means_A - X_means_B
    Y_diffs = Y_means_A - Y_means_B

    # Form joint distribution of associations
    joint_diffs = torch.cat([X_diffs, Y_diffs])

    # Compute effect size - difference in means of associations divided by standard deviation of joint distribution
    X_mean = X_diffs.mean()
    Y_mean = Y_diffs.mean()

    # Note that we use the unbiased estimator of standard deviation, which divides by N-1 instead of N (correction=1)
    joint_std = joint_diffs.std(correction=1)
    effect_size = (X_mean - Y_mean) / joint_std

    # Compute one-tailed p-value by permuting the target word associations and obtaining the difference in means for each permutation
    p_value = compute_p_value(X_diffs, Y_diffs, permutations)

    return effect_size.item(), p_value

# Standard SC-EAT
def SC_EAT(A: torch.tensor, 
            B: torch.tensor, 
            w: torch.tensor, 
            permutations:int=1000,
            ) -> tuple[float, float]:
    """
    Computes the effect size and one-tailed p-value of the SC-EAT test for the given sets of attribute (A, B) and target (w) word.
    """

    # Normalize all vectors to unit length by dividing by the norm
    A_normed = A / A.norm(dim=1, keepdim=True)
    B_normed = B / B.norm(dim=1, keepdim=True)
    w_normed = w / w.norm(dim=1, keepdim=True)

    # Compute dot product (cosine similarity, because vectors are normalized) of each target word with each attribute word
    A_w = torch.mv(A_normed, w_normed.t())
    B_w = torch.mv(B_normed, w_normed.t())

    # Form joint distribution of associations - associations are cosine similarities in SC-WEAT
    joint_associations = torch.cat([A_w, B_w])

    # Compute effect size - difference in means of associations divided by standard deviation of joint distribution
    A_mean = A_w.mean()
    B_mean = B_w.mean()

    # Note that we use the unbiased estimator of standard deviation, which divides by N-1 instead of N (correction=1)
    joint_std = joint_associations.std(correction=1)
    effect_size = (A_mean - B_mean) / joint_std

    # Compute one-tailed p-value by permuting the target word associations and obtaining the difference in means for each permutation
    p_value = compute_p_value(A_w, B_w, permutations)

    return effect_size.item(), p_value

# Single-Target EAT - note that X is used to denote the T Target word set, to prevent confusion with the transpose operator
def ST_EAT(A: torch.tensor, 
           B: torch.tensor, 
           X: torch.tensor, 
           permutations: int=1000,
           ) -> tuple[float, float]:
    """
    Computes the effect size and one-tailed p-value of the ST-EAT test for the given sets of attribute (A, B) and target (X) words.
    """

    # Normalize all vectors to unit length by dividing by the norm
    A_normed = A / A.norm(dim=1, keepdim=True)
    B_normed = B / B.norm(dim=1, keepdim=True)
    X_normed = X / X.norm(dim=1, keepdim=True)

    # Compute dot product (cosine similarity, because vectors are normalized) of each target word with each attribute word
    X_A = torch.mm(X_normed, A_normed.t())
    X_B = torch.mm(X_normed, B_normed.t())

    # Compute means, corresponding to the mean similarity of each attribute word with all target words in X
    st_means_AX = X_A.mean(dim=0)
    st_means_BX = X_B.mean(dim=0)
    joint_X_means = torch.cat([st_means_AX, st_means_BX])

    # Compute ST-EAT effect size and p-value for X
    effect_size = (st_means_AX.mean() - st_means_BX.mean()) / joint_X_means.std(correction=1)
    p_value = compute_p_value(st_means_AX, st_means_BX, permutations)

    return effect_size.item(), p_value