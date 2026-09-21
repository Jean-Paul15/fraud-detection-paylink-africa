"""
Bandit Contextuel pour seuillage dynamique en detection de fraude.
Thompson Sampling : chaque "bras" = un seuil de decision.
Le contexte (features de la transaction) determine quel seuil est optimal.

Utilisation typique:
    bandit = ThompsonSamplingBandit(thresholds=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7])
    threshold, arm = bandit.select_threshold(rng)
    # ... evaluer si la transaction est correctement classee ...
    bandit.update(arm, reward)
"""
import numpy as np


class ThompsonSamplingBandit:
    """Bandit contextuel avec Thompson Sampling pour seuil optimal.

    Pour chaque transaction, le bandit choisit un seuil parmi K possibilites.
    Apres feedback (vraie fraude ou pas), il met a jour ses croyances.

    La recompense est 1 si le seuil choisi classe correctement la transaction
    (fraude au-dessus du seuil, legitime en-dessous), 0 sinon.

    Parameters
    ----------
    thresholds : list, optional
        Seuils candidats. Defaut: [0.05, 0.1, 0.2, 0.3, 0.5, 0.7].
    prior_alpha : float, optional
        Parametre alpha du prior Beta(alpha, beta). Defaut: 1.
    prior_beta : float, optional
        Parametre beta du prior Beta(alpha, beta). Defaut: 1.
    """

    def __init__(self, thresholds=None, prior_alpha=1, prior_beta=1):
        if thresholds is None:
            self.thresholds = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
        else:
            self.thresholds = list(thresholds)

        self.K = len(self.thresholds)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

        # Compteurs par bras initialises avec le prior Beta(alpha, beta)
        self.successes = np.ones(self.K) * prior_alpha
        self.failures = np.ones(self.K) * prior_beta

    def select_threshold(self, rng=None):
        """Echantillonne un seuil selon Thompson Sampling.

        Pour chaque bras k, tire theta_k ~ Beta(successes[k], failures[k]).
        Choisit le bras avec le plus grand theta_k (probabilite de succes estimee).

        Parameters
        ----------
        rng : numpy.random.RandomState, optional
            Generateur aleatoire pour reproductibilite.

        Returns
        -------
        tuple (float, int)
            Seuil selectionne et index du bras correspondant.
        """
        if rng is None:
            rng = np.random.RandomState()

        samples = rng.beta(self.successes, self.failures)
        best_arm = int(np.argmax(samples))
        return self.thresholds[best_arm], best_arm

    def update(self, arm_idx, reward):
        """Met a jour les croyances apres observation du feedback.

        Parameters
        ----------
        arm_idx : int
            Index du bras utilise.
        reward : float
            Recompense observee: 1 si la transaction a ete correctement classee
            (fraude au-dessus du seuil, legitime en-dessous), 0 sinon.
        """
        if reward > 0:
            self.successes[arm_idx] += 1
        else:
            self.failures[arm_idx] += 1

    def get_best_threshold(self):
        """Retourne le seuil avec la meilleure probabilite de succes estimee."""
        probas = self.successes / (self.successes + self.failures)
        best_idx = int(np.argmax(probas))
        return self.thresholds[best_idx]

    def get_posterior_stats(self):
        """Retourne les statistiques actuelles par bras (pour monitoring)."""

        def _safe_list(arr):
            if hasattr(arr, "tolist"):
                return arr.tolist()
            return list(arr)

        return {
            "thresholds": list(self.thresholds),
            "successes": _safe_list(self.successes),
            "failures": _safe_list(self.failures),
            "estimated_proba": _safe_list(
                self.successes / (self.successes + self.failures)
            ),
        }
