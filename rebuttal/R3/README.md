# Time Complexity Analysis

We demonstrate that our approach has the same time-complexity, $O((|V|+|E|)*D^2)$ for each graph convolutional layer (For simpification, we assume that the dimension of input feature and hidden feature for each model layer all equal to $D$.

## Notations

First we give the definitions of each notations used in the formula:

|  Notation   |                         Description                          |
| :---------: | :----------------------------------------------------------: |
|     $V$     |                           Node set                           |
|     $E$     |                           Edge set                           |
|    $|V|$    |                       Number of nodes                        |
|    $|E|$    |    Number of edges ($|E|=|E^{v-s}|+|E^{v-v}|+|E^{s-s}|$)     |
|     $D$     |       Input feature dimension/hidden feature dimension       |
| $|E^{v-s}|$ | Number of cross-domain edges between vocal nodes and silent nodes |
| $|E^{v-v}|$ |      Number of within-domain edges between vocal nodes       |
| $|E^{s-s}|$ |      Number of within-domain edges between silent nodes      |



## Time Complexity Calculation

 Our KTGNN are composed of three main components (i.e., DAFC, DAMP, DTC), and we analyze the time complexity of each component respectively:

* Complexity of  DAFC: The time complexity of DAFC module is cuased by the cost of calculating **Domain Calibrated Variable** and **Neighbor Importance Factor** for each edge in $E^{v-s}\cup E^{s-s}$ . The calculation of Domain Calibrated Variable costs $O(D+D^2 + 2D^2)=O(D^2)$; The calculation of Neighbor Importance Factor costs $O(2D*D)=O(D^2)$. Considering that **each edge in $E^{v-s}\cup E^{s-s}$ will be used for knowledge transfer in DAFC for single time**, the final time complexity of DAFC is :

  ​																	$O\left((|E^{v-s}|+|E^{s-s}|)\cdot D^2\right)$

* Complexity of one layer in DAMP: Each layer of the DAMP module has the same time complexity to calculate the Domain Calibrated Factor and Neighbor Importance Factor as the DAFC module. Differently, DAMP need to calculate the two factors for each edge in $E$, thus the final time complexity of  one layer in DAMP is:

  ​													      				       $O\left(|E|\cdot D^2\right)$

* Complexity of DTC: The parametric component in this part is one shallow MLP with three linear transformation layers as well as non-linear activation functions. Therefore, the time complexity of the DTC module is:

  ​																       	       $O(|V|\cdot D^2)$

<u>Considering that $|E|=|E^{v-s}|+|E^{v-v}|+|E^{s-s}|$,  the final time complexity of our approach is the sum of time complexity for each module, which equals to $O((|V|+|E|)*D^2)$, which equals to the complexity of other mainstream GNNs(e.g., GAT).</u>

