optimizer = dict(type='Adagrad', lr=1e-3, weight_decay=0)

optim_wrapper = dict(
   type='OptimWrapper', optimizer=optimizer, clip_grad=None
)

optimizer_config = dict()
