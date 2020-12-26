# UM_SI650_Information_Retrieval_FP_Image_in_Recipe_out

## Introduction
When people travel or move to a new country, there may be many vegetables and fruits
they have never seen before. To find out how to make a dinner with the unseen
vegetables, they may need to take a picture of the ingredients, upload it to google image,
get the name of the vegetable from the retrieved results, and finally search for the
recipe. To simplify these steps, we build a search engine to help people decide what kind
of meal they would like to have based on the ingredients by only taking a picture. To
realize this, we build a CNN classifier to recognize the vegetable in the input image and
modify the returned label as a search query to generate recipes for users. The ranking
method we have approached is BM25 and NDCG@10 is applied as the evaluation metric to
our search engine.

More information could be accessed in this blog post: https://medium.com/information-retrieval/image-in-recipe-out-7c5198d8093f
