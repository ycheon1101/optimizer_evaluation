optimizer = dict(
    type='RMSprop', lr=1e-4, weight_decay=1e-5, momentum=0.9, alpha=0.99
)

optim_wrapper = dict(
   type='OptimWrapper', optimizer=optimizer, clip_grad=None
)

optimizer_config = dict()
