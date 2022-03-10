import matplotlib.pyplot as plt
import corner

# line
def line_diag(sampler, labels):
	shap = sampler.shape
	ndim = shap[1]
	fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
	samples = sampler.get_chain()
	for i in range(ndim):
	    ax = axes[i]
	    ax.plot(samples[:, :, i], "k", alpha=0.3)
	    ax.set_xlim(0, len(samples))
	    ax.set_ylabel(labels[i])
	    ax.yaxis.set_label_coords(-0.1, 0.5)
	axes[-1].set_xlabel("step number");

# correlation diag
def cor_diag(sampler, labels, disd = 0, thin = 1, x_inital = 0):
    flat_samples = sampler.get_chain(discard=disd, flat=True, thin=thin)
    if x_inital == 0:
        fig = corner.corner( # by defualt, 2d-hist is 1,1.5,2 sigma contour
            flat_samples,    # unless use levels=[.1,.2,.3], for example.
            labels=labels,
            );
    else:
        fig = corner.corner(
            flat_samples,
            labels=labels,
            truths=x_inital
            );

# hist diag
def hist_diag(sampler, label, channel, bins = 100, disd = 0, thin = 1):
    flat_samples = sampler.get_chain(flat=True, discard = disd, thin = thin)
    plt.hist(flat_samples[:,channel], bins, color="k", histtype="step")
    plt.xlabel(label)
    plt.ylabel("bin number")
    plt.gca().set_yticks([]);
