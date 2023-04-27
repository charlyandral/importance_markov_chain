#%%
import numpy as np
import jax
import jax.numpy as jnp
from flowMC.nfmodel.rqSpline import RQSpline
from flowMC.sampler.MALA import MALA
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.nfmodel.utils import *
import imc_functions as imc
from jax.scipy.special import logsumexp

import matplotlib.pyplot as plt
import pandas as pd
import arviz as az
import numpy.random as npr
from tqdm.notebook import trange, tqdm
import seaborn as sns



jax.config.update("jax_enable_x64", True) 
#%%
def apply_test_functions_weighted(functions, array, weights, label):
    out = {}
    if label != "":
        label = "_" + label
    for key, function in functions.items():
        out[key + label] = function(array, weights)
    return out


def compute_statistics(chain, copies, functions, label, dict_in):
    dim = chain.shape[1]
    ess_bulk = az.ess(az.convert_to_dataset(chain[np.newaxis, :])).x.to_numpy()
    ess_tail = az.ess(
        az.convert_to_dataset(chain[np.newaxis, :]), method="tail"
    ).x.to_numpy()
    statistics = apply_test_functions_weighted(functions, chain, copies, "")
    dict_out = (
        dict_in
        | {"kind": label}
        | statistics
        | {f"ess_bulk_{i}": ess_bulk[i] for i in range(dim)}
        | {f"ess_tail_{i}": ess_tail[i] for i in range(dim)}
    )
    return dict_out


def compute_statistics_weighted(chain, weights, functions, label, dict_in):
    statistics = apply_test_functions_weighted(functions, chain, weights, "")
    dict_out = dict_in | {"kind": label} | statistics
    return dict_out


def run(seed: int, n_dim: int, verbose=False):
    @jax.jit
    def target_dual_moon(x):
        """
        Term 2 and 3 separate the distribution and smear it along the first and second dimension
        """
        term1 = 0.5 * ((jnp.linalg.norm(x) - 2) / 0.1) ** 2
        terms = []
        for i in range(n_dim):
            terms.append(-0.5 * ((x[i : i + 1] + jnp.array([-3.0, 3.0])) / 0.6) ** 2)
        return -(term1 - sum([logsumexp(i) for i in terms]))

    n_chains = 50

    rng_key_set = initialize_rng_keys(n_chains, seed=seed)

    initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1

    model = RQSpline(n_dim, 10, [128, 128], 8)

    step_size = 1e-1
    MALA_Sampler = MALA(target_dual_moon, True, {"step_size": step_size})
    local_sampler_caller = lambda x: MALA_Sampler.make_sampler()

    n_loop_training = 30
    n_loop_production = 0
    n_local_steps = 10
    n_global_steps = 10
    num_epochs = 1

    learning_rate = 0.0008
    momentum = 0.9
    batch_size = 5000


    nf_sampler = Sampler(
        n_dim,
        rng_key_set,
        MALA_Sampler,
        target_dual_moon,
        model,
        n_loop_training=n_loop_training,
        n_loop_production=n_loop_production,
        n_local_steps=n_local_steps,
        n_global_steps=n_global_steps,
        n_chains=n_chains,
        n_epochs=num_epochs,
        learning_rate=learning_rate,
        momentum=momentum,
        batch_size=batch_size,
        use_global=True,
    )
    nf_sampler.sample(initial_position)

    def do_plot_train():
        out_train = nf_sampler.get_sampler_state(training=True)

        chains = np.array(out_train["chains"])
        global_accs = np.array(out_train["global_accs"])
        local_accs = np.array(out_train["local_accs"])
        loss_vals = np.array(out_train["loss_vals"])
        nf_samples = np.array(nf_sampler.sample_flow(1000)[1])

        # Plot 2 chains in the plane of 2 coordinates for first visual check
        plt.figure(figsize=(6, 6))
        axs = [plt.subplot(2, 2, i + 1) for i in range(4)]
        plt.sca(axs[0])
        plt.title("2d proj of 2 chains")

        plt.plot(chains[0, :, 0], chains[0, :, 1], "o-", alpha=0.5, ms=2)
        plt.plot(chains[1, :, 0], chains[1, :, 1], "o-", alpha=0.5, ms=2)
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")

        plt.sca(axs[1])
        plt.title("NF loss")
        plt.plot(loss_vals.reshape(-1))
        plt.xlabel("iteration")

        plt.sca(axs[2])
        plt.title("Local Acceptance")
        plt.plot(local_accs.mean(0))
        plt.xlabel("iteration")

        plt.sca(axs[3])
        plt.title("Global Acceptance")
        plt.plot(global_accs.mean(0))
        plt.xlabel("iteration")
        plt.tight_layout()
        plt.show(block=False)

        labels = ["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$"]

    # Plot all chains
    # figure = corner.corner(
    # chains.reshape(-1, n_dim))#, labels=labels

    # figure.set_size_inches(7, 7)
    # figure.suptitle("Visualize samples")
    # plt.show(block=False)

    # Plot Nf samples
    # figure = corner.corner(nf_samples)#, labels=labels)
    # figure.set_size_inches(7, 7)
    # figure.suptitle("Visualize NF samples")
    # plt.show()

    if verbose:
        do_plot_train()
    return target_dual_moon, nf_sampler


#%%


#%%


def do_imh_imc(
    length: int,
    dims: np.array(int),
    seed: int,
    test_functions: dict,
    test_functions_is: dict,
    n_flow: int = 5,
    n_rep: int = 10,
    pseudo_marginal: bool = False,
    var_pseudo_marginal: float = 1,
    alphas=[1],
    save=True,
):
    rows = []
    # initiate the numpy seed that will determine all the other
    npr.seed(seed)

    # initiate the numba seed
    seed_numba = npr.randint(0, 1e6)
    imc.seed_numba(seed_numba)

    # seeds for the flows, one per flow
    seeds = npr.randint(0, 1e6, size=n_flow)

    # iteration on the list on dimensions to repeat
    for dim in tqdm(dims, desc="dim"):
        # iteration on the flow
        for seed_flow in tqdm(seeds, desc="flow"):
            # make a unique seed for this couple (dimension,flow)
            print(seed_flow)
            local_seed = int(seed_flow + dim * 1e7)
            print(local_seed)
            target_dual_moon, nf_sampler= run(
                local_seed, dim, verbose=False
            )
            evaluate_flow_jit = jax.jit(nf_sampler.evalulate_flow)
            # do n_rep for this flow
            for i in trange(n_rep, desc="rep"):
                out_nf = nf_sampler.sample_flow(length)[1]

                # random state modification  of  nf_sampler
                nf_sampler.rng_keys_nf += 1

                # evaluation of the log density of target
                target_density = jnp.apply_along_axis(target_dual_moon, 1, out_nf)

                # evaluation of the log density of instrumental
                instrumental_density = evaluate_flow_jit(out_nf)

                # compute the importance weights
                weights = jnp.exp(target_density - instrumental_density)
                # if pseudo marginal, noise added to the weights
                if pseudo_marginal:
                    weights *= npr.gamma(
                        1 / var_pseudo_marginal, var_pseudo_marginal, size=length
                    )
                dict_out = {
                        "length_chain": length,
                        "dimension": dim,
                        "seed_train": seed,
                        "seed_gen": local_seed,
                        "seed_rep": i,
                    }
                for alpha in alphas:
                    
                    # compute the importance chain
                    out_imc, copies_imc = imc.compute_chain(
                        np.array(out_nf), np.array(weights), alpha=alpha
                    )
                    rows.append(
                        compute_statistics(
                            out_imc, copies_imc, test_functions, "imc", dict_out | {"alpha": alpha}
                        )
                    )

                # compute the MH chain
                out_imh, copies_imh = imc.indep_MH_nf(np.array(out_nf), np.array(weights))

                rows.append(
                    compute_statistics(out_imh,copies_imh,test_functions, "imh", dict_out)
                )

                #compute the OSR chain
                out_osr, copies_osr = imc.compute_chain(np.array(out_nf), np.array(weights),kind="osr")
                rows.append(
                    compute_statistics(out_osr,copies_osr,test_functions, "osr", dict_out)
                )

                # compute the Importance sampling estimate
                rows.append(
                    compute_statistics_weighted(
                        out_nf, weights, test_functions_is, "is", dict_out
                    )
                )

    output = pd.DataFrame(rows)
    if save:
        output.to_pickle(
            f"results_{seed}_nrep_{n_rep}_pm_{var_pseudo_marginal if pseudo_marginal else pseudo_marginal}.pkl.zip"
        )
    return pd.DataFrame(rows)


rows_test = do_imh_imc(
    30000,
    dims=[5,10,15,20,25],
    seed=1250,
    n_flow=10,
    n_rep=30,
    pseudo_marginal=False,
    var_pseudo_marginal=1,
    alphas=[1],
    test_functions={
        "mean": lambda array, copies: float(np.mean(array)),
        "mean_1": lambda array, copies: float(np.mean(array[:, 0])),
        "third": lambda array, copies: float(np.mean(array**3)),
        "third_1": lambda array, copies: float(np.mean(array[:, 0] ** 3)),
        "fifth_1": lambda array, copies: float(np.mean(array[:, 0] ** 5)),
        "fifth": lambda array, copies: float(np.mean(array**5)),
        "seventh": lambda array, copies: float(np.mean(array**7)),
        "seventh_1": lambda array, copies: float(np.mean(array[:, 0] ** 7)),
        "norm": lambda array, copies: float(np.mean(np.linalg.norm(array, axis=1))),
        "sjd": lambda array, copies: imc.JMP(array),
        "postive_copies": lambda array, copies: np.sum(copies > 0),
        "ess_is": lambda array, copies: imc.ESS_IS(copies),
        "length": lambda array, copies: len(array),
    },
    test_functions_is={
        "mean": lambda array, weights: float(
            np.average(np.mean(array, axis=1), weights=weights)
        ),
        "mean_1": lambda array, weights: float(
            np.average(array[:, 0], weights=weights)
        ),
        "third": lambda array, weights: float(
            np.average(np.mean(array**3, axis=1), weights=weights)
        ),
        "third_1": lambda array, weights: float(
            np.average(array[:, 0] ** 3, weights=weights)
        ),
        "fifth": lambda array, weights: float(
            np.average(np.mean(array**5, axis=1), weights=weights)
        ),
        "fifth_1": lambda array, weights: float(
            np.average(array[:, 0] ** 5, weights=weights)
        ),
        "seventh": lambda array, weights: float(
            np.average(np.mean(array**7, axis=1), weights=weights)
        ),
        "seventh_1": lambda array, weights: float(
            np.average(array[:, 0] ** 7, weights=weights)
        ),
        "norm": lambda array, weights: float(
            np.average(np.linalg.norm(array, axis=1), weights=weights)
        ),
        "ess_is": lambda array, weights: float(imc.ESS_IS(np.array(weights)))
    },
    save=True,
)


# %%

fig = sns.boxplot(x="dimension",y = "ess_bulk_1",hue = "kind",data = rows_test[rows_test.kind != "is"])
fig.set_ylabel("ESS")
fig.get_legend().set_title("")
fig.get_figure().savefig("normalizing_flow_ess.pdf")
# %%
rows_test.loc[rows_test.kind =="is","postive_copies"] = rows_test[rows_test.kind =="is"].length_chain
mse = lambda serie : np.mean(serie**2)
mse_table = rows_test[rows_test.kind.isin(['imc','is'])].groupby(by = ["dimension","kind"])[["mean_1","third_1","fifth_1","seventh_1"]].aggregate(mse)

mse_table["positive_copies"] = rows_test[rows_test.kind.isin(['imc','is'])].groupby(by = ["dimension","kind"])["postive_copies"].apply(np.mean)

print(mse_table.style.format("{:.3e}").to_latex())
