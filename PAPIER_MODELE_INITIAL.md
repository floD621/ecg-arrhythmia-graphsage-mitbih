# Classification binaire des arythmies ECG par GraphSAGE sur des graphes de battements

## Résumé

La détection automatique des arythmies cardiaques à partir de signaux électrocardiographiques (ECG) constitue un problème important pour l'aide au dépistage et l'analyse à grande échelle des enregistrements. Nous présentons une méthode de classification binaire appliquée à la base MIT-BIH Arrhythmia Database. Chaque enregistrement est transformé en une suite de battements annotés, puis découpé en fenêtres temporelles. Une fenêtre devient un graphe dont les nœuds représentent les battements et dont les arêtes relient les battements temporellement adjacents. Les nœuds sont décrits par des statistiques morphologiques simples et par les intervalles RR. Un réseau GraphSAGE agrège l'information locale avant un pooling moyen global et une classification binaire.

Sur l'évaluation rapportée, le modèle obtient une accuracy de **0,975824** et un F1-macro de **0,975254** sur 4 674 graphes. La matrice de confusion est `[[2635, 47], [66, 1926]]`. Ces résultats montrent une excellente performance apparente sur la partition utilisée. Ils doivent toutefois être interprétés avec prudence, car le protocole repose sur un split aléatoire de segments, alors que les fenêtres d'un même enregistrement se recouvrent et peuvent donc partager une information très proche entre apprentissage et évaluation.

**Mots-clés :** ECG, MIT-BIH, arythmie, graphes, GraphSAGE, apprentissage profond.

## 1. Introduction

Les enregistrements ECG contiennent à la fois une information morphologique, portée par la forme de chaque battement, et une information temporelle, portée notamment par les intervalles entre battements. Une arythmie peut ainsi se manifester par une forme de complexe atypique, par une irrégularité du rythme, ou par la combinaison des deux phénomènes.

Les réseaux convolutifs et récurrents traitent naturellement les signaux sous forme de séquences ou de fenêtres régulières. Une représentation en graphe offre une autre manière de modéliser le problème : chaque battement devient un nœud et les relations temporelles entre battements sont représentées explicitement par des arêtes. Cette représentation permet d'appliquer une agrégation de voisinage, de manière à ce que la représentation d'un battement dépende également de son contexte immédiat.

L'objectif de ce travail est d'évaluer un premier modèle GraphSAGE simple et interprétable pour distinguer deux catégories de fenêtres ECG : `normal` et `arrhythmia`. Le modèle constitue une première base expérimentale avant l'étude de variantes plus complexes.

## 2. Matériel et données

### 2.1 Base MIT-BIH Arrhythmia Database

L'étude utilise la MIT-BIH Arrhythmia Database, une base de référence distribuée par PhysioNet. Elle contient 48 enregistrements ECG annotés battement par battement, échantillonnés à 360 Hz. Les fichiers d'annotation WFDB fournissent notamment la position de chaque annotation et son symbole clinique.

Le pipeline lit, pour chaque enregistrement, le signal ECG et les annotations correspondantes. Le premier canal du signal est utilisé. Les battements sont extraits autour de la position annotée : 0,25 seconde avant l'annotation et 0,45 seconde après, soit une fenêtre nominale de 0,70 seconde à 360 Hz. Les battements trop proches des limites du signal sont ignorés.

### 2.2 Définition des classes

Les annotations sont regroupées en deux classes. La classe `normal` regroupe les battements considérés comme normaux dans le codage utilisé par le pipeline. La classe `arrhythmia` regroupe une fenêtre dès qu'elle contient au moins un symbole associé à une activité atriale, ventriculaire, de fusion ou à un rythme pathologique.

Le label est donc attribué au niveau de la fenêtre et non au niveau d'un battement isolé : une fenêtre contenant au moins un battement pathologique est étiquetée `arrhythmia`. Ce choix privilégie la sensibilité à la présence d'une anomalie dans le segment, mais signifie également qu'une fenêtre peut contenir à la fois plusieurs battements normaux et un seul battement pathologique.

### 2.3 Segmentation

La séquence de battements de chaque enregistrement est découpée par fenêtres glissantes de 12 battements, avec un pas de 6 battements. Deux fenêtres successives partagent donc généralement 6 battements. Cette stratégie augmente le nombre d'exemples et conserve le contexte local, mais elle crée une forte dépendance entre exemples voisins.

## 3. Construction des graphes

### 3.1 Nœuds

Chaque fenêtre de 12 battements est représentée par un graphe comportant jusqu'à 12 nœuds. Un nœud correspond à un battement annoté. Les informations utilisées pour construire ses features sont la forme locale du battement, sa durée et son contexte temporel.

### 3.2 Features de battement

Le vecteur de features comporte 10 composantes numériques :

1. moyenne de l'amplitude du battement ;
2. écart-type ;
3. minimum ;
4. maximum ;
5. étendue, égale au maximum moins le minimum ;
6. moyenne de la valeur absolue du signal recentré ;
7. valeur quadratique moyenne du signal recentré ;
8. durée de l'extrait ;
9. intervalle RR précédent ;
10. intervalle RR suivant.

Les intervalles RR sont exprimés en secondes. Pour le premier ou le dernier battement d'un enregistrement, la valeur manquante est remplacée par zéro. Les features sont donc compactes et peu coûteuses à calculer. En revanche, elles ne conservent pas directement toute la morphologie échantillon par échantillon de l'onde ECG.

### 3.3 Arêtes

Les arêtes encodent le voisinage temporel immédiat. Pour chaque battement, une arête est créée vers le battement précédent et vers le battement suivant lorsqu'ils existent. Le graphe est donc une chaîne bidirectionnelle de battements. Il n'y a pas d'arêtes basées sur une similarité morphologique ni de connexion directe entre battements éloignés.

## 4. Architecture du modèle

### 4.1 Propagation GraphSAGE

Le modèle `GraphSAGEModel` reçoit la matrice des features des nœuds, l'index des arêtes et l'identifiant de graphe de chaque nœud dans un mini-lot. Il comprend deux couches `SAGEConv` :

- la première projette les 10 features d'entrée vers un espace latent de dimension 128 ;
- la seconde conserve cette dimension de 128 et agrège à nouveau les représentations du voisinage.

Après chaque couche, une activation ReLU est appliquée. Pour un nœud donné, la convolution GraphSAGE combine sa représentation avec une agrégation des représentations de ses voisins. La représentation finale de chaque nœud contient ainsi des informations provenant de son battement et de son voisinage temporel à plusieurs degrés.

### 4.2 Pooling et classification

Après les deux couches de message passing, un pooling moyen global agrège les représentations des nœuds d'un même graphe. Pour un graphe $G$ contenant $N$ nœuds et des représentations $h_i$, l'embedding global est calculé comme suit :

$$
h_G = \frac{1}{N}\sum_{i=1}^{N} h_i.
$$

Cet embedding de dimension 128 est ensuite traité par une petite tête de projection composée d'une couche linéaire vers 128 dimensions et d'une activation ReLU. La classification finale est réalisée par une couche linéaire indépendante qui produit deux logits, correspondant aux classes `normal` et `arrhythmia`.

Le pooling moyen rend le modèle invariant à l'ordre de présentation des nœuds dans le mini-lot, tout en laissant les convolutions GraphSAGE exploiter la structure temporelle avant l'agrégation. En contrepartie, il résume tous les battements par une moyenne et peut atténuer l'influence d'un battement rare mais déterminant.

### 4.3 Paramètres d'entraînement

L'entraînement utilise :

- l'optimiseur Adam ;
- un taux d'apprentissage de $10^{-3}$ ;
- une pénalisation L2 (`weight_decay`) de $10^{-4}$ ;
- une entropie croisée comme fonction de coût ;
- des mini-lots de 16 graphes ;
- une seed fixée à 42 pour les générateurs utilisés ;
- une sélection du meilleur checkpoint selon le F1-macro de validation.

Le modèle est entraîné pendant 100 epochs dans l'expérience rapportée. Le checkpoint est sauvegardé dans `weights/gnn_best.pth` lorsqu'il améliore le F1-macro observé sur la partition de validation.

## 5. Protocole expérimental

Les graphes sont séparés en une partition d'apprentissage représentant 75 % des exemples et une partition de test/validation représentant 25 %. La séparation est effectuée avec `train_test_split`, avec stratification selon le label binaire et une seed égale à 42.

Le protocole est réalisé au niveau des graphes, après génération des fenêtres glissantes. Il ne garantit donc pas que deux fenêtres provenant du même enregistrement restent dans une seule partition. Cette précision est importante pour interpréter les résultats, car les fenêtres voisines se recouvrent de moitié.

Pendant chaque epoch, les paramètres sont optimisés sur les graphes d'apprentissage. Le modèle est ensuite évalué sur les graphes de validation sans mise à jour des poids. L'accuracy et le F1-macro sont calculés, et le meilleur état selon le F1-macro est conservé.

## 6. Résultats

### 6.1 Performances globales

L'évaluation du meilleur checkpoint fournit les résultats suivants :

| Métrique | Valeur |
| --- | ---: |
| Accuracy | 0,975824 |
| F1-macro | 0,975254 |
| Nombre de graphes évalués | 4 674 |
| Epoch du meilleur checkpoint | 97 |

La matrice de confusion, avec les lignes correspondant aux classes réelles et les colonnes aux classes prédites, est :

$$
\begin{pmatrix}
2635 & 47 \\
66 & 1926
\end{pmatrix}.
$$

Cela correspond à 2 635 fenêtres normales correctement reconnues, 47 fenêtres normales classées à tort comme arythmiques, 66 fenêtres arythmiques classées à tort comme normales et 1 926 fenêtres arythmiques correctement reconnues.

### 6.2 Interprétation des résultats

L'accuracy élevée indique que la grande majorité des fenêtres sont correctement classées. Le F1-macro très proche de l'accuracy suggère que la performance ne repose pas uniquement sur une classe majoritaire : les deux classes obtiennent des performances globalement équilibrées selon le protocole utilisé.

La matrice de confusion montre cependant une asymétrie entre les erreurs. Le modèle produit 66 faux négatifs pour la classe arythmie, contre 47 faux positifs. Dans un contexte de dépistage, les faux négatifs sont particulièrement importants, car ils correspondent à des fenêtres contenant une anomalie mais déclarées normales par le modèle. Le résultat est donc favorable, tout en justifiant une lecture distincte de l'accuracy et du F1-macro.

Il faut également distinguer la performance de classification de la portée clinique. Les chiffres mesurent la capacité du pipeline à reproduire le label de fenêtre défini à partir des annotations MIT-BIH. Ils ne constituent pas une mesure directe de la sécurité d'un système médical déployé ni de sa capacité à généraliser à de nouveaux patients.

## 7. Discussion critique

### 7.1 Points forts

**Représentation structurée du rythme.** Le graphe encode explicitement la succession des battements. Le modèle ne traite donc pas chaque battement comme une observation indépendante : les convolutions propagent l'information entre battements voisins.

**Architecture compacte.** Deux couches GraphSAGE, un pooling moyen et une tête linéaire forment un modèle relativement simple à entraîner et à inspecter. Cette simplicité facilite la reproduction de l'expérience et limite le nombre de choix architecturaux.

**Features interprétables.** Les statistiques d'amplitude, la durée et les intervalles RR ont une signification physiologique claire. Elles permettent de relier une partie du comportement du modèle à des caractéristiques observables du signal.

**Résultats équilibrés sur la partition rapportée.** La combinaison de l'accuracy, du F1-macro et de la matrice de confusion donne une vision plus utile qu'une seule métrique. Le nombre de faux négatifs est explicitement mesurable.

### 7.2 Limites expérimentales

**Risque de fuite entre partitions.** C'est la limite principale. Les fenêtres sont générées avec un recouvrement de 50 %, puis réparties aléatoirement. Deux graphes placés dans des partitions différentes peuvent donc partager plusieurs battements identiques ou presque identiques et appartenir au même enregistrement. La performance mesurée peut ainsi être plus optimiste qu'une évaluation sur des enregistrements ou des patients entièrement non vus.

**Validation au niveau des segments.** Le protocole ne fournit pas une estimation stricte de la généralisation à un nouveau patient. La présence d'un même enregistrement dans les deux partitions réduit l'indépendance statistique des observations.

**Features morphologiques réduites.** Les dix statistiques résument fortement la forme du battement. Deux battements de morphologies différentes peuvent produire des résumés proches. Le modèle n'exploite pas directement les échantillons de l'onde, ni une représentation apprise de la morphologie complète.

**Label de fenêtre peu granulaire.** Une fenêtre est positive dès qu'un seul battement est associé à une arythmie. Le modèle apprend donc à détecter la présence d'au moins un événement dans une fenêtre, mais il n'est pas entraîné à localiser précisément le battement responsable.

**Métriques incomplètes.** L'expérience rapportée ne fournit pas l'AUROC, l'AUPRC, la sensibilité et la spécificité séparées, ni d'intervalle de confiance. L'accuracy et le F1-macro sont utiles, mais insuffisants pour caractériser complètement une tâche potentiellement déséquilibrée et orientée vers le dépistage.

**Une seule exécution.** Les résultats sont rapportés pour une seed et une partition données. Ils ne permettent pas d'estimer la variabilité du modèle selon la partition, la seed ou la composition des enregistrements.

### 7.3 Bilan scientifique

Le premier modèle valide la faisabilité d'une représentation en graphes temporels de battements ECG et fournit une base solide pour une étude comparative. Les scores élevés et la matrice de confusion montrent que le pipeline apprend effectivement une séparation utile sur les graphes évalués.

Cependant, le protocole ne permet pas de conclure que le modèle conserverait ces performances sur des patients absents de l'apprentissage. La principale conclusion doit donc rester méthodologique : GraphSAGE peut exploiter des features compactes de battements et leurs relations temporelles pour classifier des fenêtres MIT-BIH, mais l'ampleur réelle de la généralisation nécessite une évaluation indépendante au niveau des enregistrements ou des patients.

## 8. Conclusion

Nous avons présenté un modèle GraphSAGE pour la classification binaire de fenêtres ECG représentées comme des graphes de battements. Le pipeline utilise des annotations MIT-BIH, des fenêtres de 12 battements, des arêtes temporelles entre voisins immédiats et dix features statistiques par nœud. Le modèle applique deux couches GraphSAGE, un pooling moyen global et une classification linéaire.

Sur la partition expérimentale disponible, le meilleur checkpoint atteint une accuracy de 0,975824 et un F1-macro de 0,975254 sur 4 674 graphes. La matrice de confusion contient 66 faux négatifs arythmiques et 47 faux positifs. Ces résultats sont encourageants pour une première architecture, mais leur interprétation doit tenir compte du split aléatoire au niveau des segments et du recouvrement important entre fenêtres. La contribution principale de cette expérience est ainsi la démonstration d'un pipeline GNN cohérent et performant sur la partition évaluée, avec une limite de validité externe clairement identifiée.
