# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0


# optimizer
optimizer = dict(
   type='SGD', lr=2.5e-4, momentum=0.9, weight_decay=0.0005)


optim_wrapper = dict(
   type='OptimWrapper', optimizer=optimizer, clip_grad=None
)


optimizer_config = dict()



