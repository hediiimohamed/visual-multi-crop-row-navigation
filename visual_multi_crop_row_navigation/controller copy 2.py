#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Contrôleur visuel pour la navigation dans les rangées de culture
Inspiré de la version C++ du code original
"""

import numpy as np

# ============================================================================
# CONSTANTES DU CONTRÔLEUR (comme en C++)
# ============================================================================

# Gains pour la marche avant
LAMBDA_X_1 = 10.0      # Gain translation
LAMBDA_W_1 = 3000.0    # Gain rotation

# Gains pour la marche arrière
LAMBDA_X_2 = 5.0       # Gain translation (marche arrière)
LAMBDA_W_2 = 1500.0    # Gain rotation (marche arrière)

# Limites de sécurité
W_MAX = 0.3            # Vitesse angulaire max (rad/s)
Z_MIN = 0.05           # Seuil minimum pour activer la rotation
MIN_POINTS = 5         # Nombre minimum de points pour activer le contrôle
V_MIN = 0.05           # Vitesse linéaire minimale pour activer la rotation

# Compteur pour les statistiques
_frame_counter = 0


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def wrapToPi(theta):
    """
    Wrap an angle to the interval [-pi, pi]
    
    Args:
        theta: angle en radians
        
    Returns:
        angle wrapped to [-pi, pi]
    """
    while theta < -np.pi:
        theta = theta + 2 * np.pi
    while theta > np.pi:
        theta = theta - 2 * np.pi
    return theta


def display_stats(err_x, err_theta, num_points, omega):
    """
    Affiche les statistiques de navigation (comme en C++)
    """
    print(f"📊 NAV STATS | err_x: {abs(err_x):.2f}px | "
          f"err_theta: {abs(err_theta):.3f}rad | "
          f"points: {num_points} | omega: {omega:.3f} rad/s")


# ============================================================================
# CONTRÔLEUR PRINCIPAL
# ============================================================================

def visualServoingCtl(camera, desiredState, actualState, v_des, 
                      steering_dir=1, num_points=10, mode=1):
    """
    Contrôleur visuel pour le suivi de rangées de culture
    
    Args:
        camera: objet Camera avec paramètres (tilt_angle, deltaz, deltay)
        desiredState: état désiré [x_des, y_des, theta_des]
        actualState: état actuel [x, y, theta]
        v_des: vitesse linéaire désirée (m/s)
        steering_dir: direction (1=avant, -1=arrière)
        num_points: nombre de points/plantes détectés
        mode: mode de navigation (1-6)
        
    Returns:
        omega: vitesse angulaire calculée (rad/s)
    """
    global _frame_counter
    
    # Extraire les états
    x = actualState[0]
    y = actualState[1]
    theta = actualState[2]
    
    # ========================================================================
    # VÉRIFICATIONS DE SÉCURITÉ (comme en C++)
    # ========================================================================
    
    # 1. Vérifier le nombre de points
    if num_points < MIN_POINTS:
        print(f"⚠️ Not enough points: {num_points} < {MIN_POINTS}")
        return 0.0
    
    # 2. Vérifier la vitesse linéaire minimale
    if abs(v_des) < V_MIN:
        return 0.0
    
    # 3. Modes spéciaux sans rotation (comme en C++)
    if mode in [2, 4]:
        return 0.0
    
    # ========================================================================
    # SÉLECTION DES GAINS
    # ========================================================================
    
    if steering_dir == 1:
        lambda_x = LAMBDA_X_1
        lambda_w = LAMBDA_W_1
    else:
        lambda_x = LAMBDA_X_2
        lambda_w = LAMBDA_W_2
    
    lambdavec = np.array([lambda_x, lambda_w])
    controller_type = 0  # 0 = translation, 2 = rotation
    
    # ========================================================================
    # PARAMÈTRES CAMÉRA
    # ========================================================================
    
    angle = camera.tilt_angle
    delta_z = camera.deltaz
    delta_y = camera.deltay
    
    # ========================================================================
    # MATRICE D'INTERACTION Ls (3x6)
    # ========================================================================
    
    # Ligne 0: pour la translation
    Ls0 = np.array([
        (-np.sin(angle) - y * np.cos(angle)) / delta_z,
        0,
        x * (np.sin(angle) + y * np.cos(angle)) / delta_z,
        x * y,
        -1 - x**2,
        y
    ])
    
    # Ligne 1: pour la translation latérale (non utilisé)
    Ls1 = np.array([
        0,
        -(np.sin(angle) + y * np.cos(angle)) / delta_z,
        y * (np.sin(angle) + y * np.cos(angle)) / delta_z,
        1 + y**2,
        -x * y,
        -x
    ])
    
    # Ligne 2: pour la rotation (theta)
    Ls2 = np.array([
        np.cos(angle) * np.cos(theta)**2 / delta_z,
        np.cos(angle) * np.cos(theta) * np.sin(theta) / delta_z,
        -(np.cos(angle) * np.cos(theta) * (y * np.sin(theta) + x * np.cos(theta))) / delta_z,
        -(y * np.sin(theta) + x * np.cos(theta)) * np.cos(theta),
        -(y * np.sin(theta) + x * np.cos(theta)) * np.sin(theta),
        -1
    ])
    
    # Matrice Ls complète (3x6)
    Ls = np.vstack([Ls0, Ls1, Ls2])
    
    # ========================================================================
    # MATRICE DE TRANSFORMATION CAMÉRA → ROBOT (cTR)
    # ========================================================================
    
    # Matrice c (6x2) comme en C++
    c = np.array([
        [0, -delta_y],
        [-np.sin(angle), 0],
        [np.cos(angle), 0],
        [0, 0],
        [0, -np.cos(angle)],
        [0, -np.sin(angle)]
    ])
    
    # Vecteurs de transformation
    Tv = c[:, 0]  # Translation (colonne 0)
    Tw = c[:, 1]  # Rotation (colonne 1)
    
    # ========================================================================
    # JACOBIENS
    # ========================================================================
    
    # Sélectionner les lignes de Ls pour le contrôleur (translation et rotation)
    J = np.vstack([Ls[controller_type, :], Ls[2, :]])  # (2x6)
    
    # Jacobien pour la vitesse linéaire
    Jv = J @ Tv  # (2x1)
    
    # Jacobien pour la vitesse angulaire
    Jw = J @ Tw  # (2x1)
    
    # ========================================================================
    # CALCUL DE L'ERREUR
    # ========================================================================
    
    # Erreur de translation
    trans_delta = actualState[controller_type] - desiredState[controller_type]
    
    # Erreur angulaire (wrap to pi)
    ang_delta = wrapToPi(actualState[2] - desiredState[2])
    
    err = np.array([trans_delta, ang_delta])
    
    # ========================================================================
    # CALCUL DE LA COMMANDE ANGULAIRE
    # ========================================================================
    
    try:
        # Calcul de la pseudo-inverse de Jw (2x1 → 1x2)
        # Pour un vecteur colonne, la pseudo-inverse est sa transposée divisée par sa norme au carré
        Jw_norm = np.linalg.norm(Jw)
        
        if Jw_norm > 1e-6:  # Éviter la division par zéro
            # Pseudo-inverse de Jw
            Jw_pinv = Jw.T / (Jw_norm ** 2)  # (1x2)
            
            # Commande angulaire
            # w = -Jw_pinv * (λ * e + Jv * v_des)
            w_cmd = -Jw_pinv @ (lambdavec * err + Jv * v_des)
            
            # Extraire la valeur scalaire
            if isinstance(w_cmd, np.ndarray):
                w = float(w_cmd[0]) if w_cmd.size > 0 else 0.0
            else:
                w = float(w_cmd)
        else:
            w = 0.0
            
    except Exception as e:
        print(f"⚠️ Control calculation error: {e}")
        return 0.0
    
    # ========================================================================
    # LIMITATION DE LA VITESSE ANGULAIRE (comme en C++)
    # ========================================================================
    
    # Limite à W_MAX
    if abs(w) > W_MAX:
        w = W_MAX if w > 0 else -W_MAX
    
    # Seuil minimum pour rotation (éviter les micro-oscillations)
    if abs(w) < Z_MIN:
        w = 0.0
    
    # Appliquer la direction (comme en C++: steering_dir * w)
    w = steering_dir * w
    
    # ========================================================================
    # STATISTIQUES (affichage périodique)
    # ========================================================================
    
    _frame_counter += 1
    if _frame_counter % 10 == 0:
        display_stats(trans_delta, ang_delta, num_points, w)
    
    return w


# ============================================================================
# CLASSE CONTRÔLEUR (optionnelle, pour une utilisation avec état)
# ============================================================================

class VisualServoingController:
    """
    Classe contrôleur avec état (maintient l'historique des erreurs)
    """
    
    def __init__(self):
        self.frame_counter = 0
        self.error_history = []
        
    def controller(self, camera, desiredState, actualState, v_des,
                   steering_dir=1, num_points=10, mode=1):
        """
        Méthode wrapper autour de la fonction principale
        """
        # Appeler la fonction principale
        omega = visualServoingCtl(camera, desiredState, actualState, v_des,
                                  steering_dir, num_points, mode)
        
        # Stocker l'erreur pour analyse
        if len(actualState) >= 2 and len(desiredState) >= 2:
            err_x = abs(actualState[0] - desiredState[0])
            self.error_history.append(err_x)
            if len(self.error_history) > 100:
                self.error_history.pop(0)
        
        self.frame_counter += 1
        return omega
    
    def get_mean_error(self):
        """Retourne l'erreur moyenne"""
        if self.error_history:
            return np.mean(self.error_history)
        return 0.0
    
    def reset(self):
        """Réinitialise le contrôleur"""
        self.frame_counter = 0
        self.error_history = []


# ============================================================================
# TEST SIMPLE (si exécuté directement)
# ============================================================================

if __name__ == "__main__":
    # Test rapide du contrôleur
    print("Testing visual servoing controller...")
    
    # Créer une caméra factice
    class DummyCamera:
        def __init__(self):
            self.tilt_angle = np.deg2rad(-80)
            self.deltaz = 1.0
            self.deltay = 0.0
    
    camera = DummyCamera()
    
    # États
    desired = np.array([0.0, 640/2, 0.0])
    actual = np.array([10.0, 640/2 + 50, 0.1])
    
    # Calcul
    omega = visualServoingCtl(camera, desired, actual, 0.2, 
                              steering_dir=1, num_points=12, mode=1)
    
    print(f"Test result: omega = {omega:.3f} rad/s")
    print("Controller is ready!")