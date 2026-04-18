import numpy as np
import matplotlib.pyplot as plt

def normal_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def lognormal_pdf(x, mu, sigma):
    return (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mu)**2) / (2 * sigma**2))

def demo_normal_lognormal(n_sample=100_000, mu=0.5, sigma=0.5):
    z = np.random.standard_normal(n_sample) 
    g = mu + sigma * z
    L = np.exp(g)
    log_L = np.log(L)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(g, bins=60, density=True, alpha=0.6, color="steelblue")
    gx = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
    axes[0].plot(gx, normal_pdf(gx, mu, sigma), "r-", lw=2, label=r"$N(\mu,\sigma^2)$")
    axes[0].set_title("Нормальное $G$")
    axes[0].grid(True, ls="--", alpha=0.5)
    axes[0].legend()

    L_hi = np.percentile(L, 99)
    Lx = np.linspace(1e-6, L_hi, 500)
    axes[1].hist(L, bins=100, range=(0, L_hi), density=True, alpha=0.6, color="seagreen")
    axes[1].plot(Lx, lognormal_pdf(Lx, mu, sigma), "r-", lw=2, label="Lognormal")
    axes[1].set_title(r"Логнормальное $L=e^G$")
    axes[1].grid(True, ls="--", alpha=0.5)
    axes[1].legend()

    axes[2].hist(log_L, bins=60, density=True, alpha=0.6, color="coral")
    axes[2].plot(gx, normal_pdf(gx, mu, sigma), "r-", lw=2, label=r"$N(\mu,\sigma^2)$")
    axes[2].set_title(r"$\ln L$")
    axes[2].grid(True, ls="--", alpha=0.5)
    axes[2].legend()

    plt.suptitle(rf"Связь распределений ($\mu={mu}$, $\sigma={sigma}$, $n={n_sample}$)", fontsize=14)
    plt.tight_layout()
    
    plt.savefig('dist_comparison.png', dpi=200)
    plt.close()

if __name__ == "__main__":
    demo_normal_lognormal()
