# LLDA
Lie Group Linear Discriminant Analysis (LLDA) is an algorithm I proposed and published in 2019: Journal of Xianyang Normal University 34(6): 24-27, 2019. An English version is also uploaded in this repository, check: [LLDA](./LLDA.pdf).

[The above publication] analyzes images' Lie group covariance features, based on the main idea of my another previous published work ([Research on Lie Group kernel learning algorithm](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=1G5TXggAAAAJ&citation_for_view=1G5TXggAAAAJ:UeHWp8X0CEIC)), we proposed LLDA (Lie group Linear Discriminant Analysis) algorithm for image classification. The main idea of this algorithm is to apply Linear Discriminant Analysis (LDA) to images' covariances, which forms a Lie group manifold, and compute a one-parameter sub group determined by a Lie algebra element and the intrinsic mean of image features. This one-parameter sub group is a geodesic on the Lie group formed by original image set. By defining the projection in Lie group, this geodesic can be calculated by the idea of LDA. Experimental results on handwritten classification show that LLDA has significantly better classification performance than some classic methods such as LDA.

## Code
The code is written in Python, and the MNIST dataset (in original data format) is included in this repo under ./dataset. To run the demo which does training and testing in a row:
Usage: python SLLDA.py <0-9> <0-9> <m> <n>

Where 
        <0-9> indicates digit number between 0 and 9 for a class
        m in <m> indicates instances for each training calss
        n in <n> indicates instances for each testing class

For instance, to do a quick experiment on classification between 1 and 0, run:
```bash
cd code
python3 SLLDA.py 1 0 200 1000
```
