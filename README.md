# torch_filter
A plug-and-play pytorch module to implement custom filtering for images.\
Sometimes, we need to implement some image filtering operations in the neural network. Therefore, we design this simple torch_filter module.
## inputs of torch_filter.py
**filter_weight**: the custom filter weights. Note that both dimensions of the filter must be equal and odd. eg. (3×3), (5×5). Type must be **np.array**.\
**is_grad**: True or False. Whether this module is involved in backpropagation.
# test.py
The test.py implements image sharpening operation.
```python
filter_weight=np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
```
![](https://github.com/deepxzy/torch_filter/blob/master/images/img.png?raw=true)
![](https://github.com/deepxzy/torch_filter/blob/master/images/new_img.png?raw=true)
