# Neural Network
Example:

model = Model()

model.add(Dense(input_dim=2, output_dim=3))
model.add(Activation(func_type='sigmoid'))
model.add(Dense(output_dim=3))
model.add(Activation(func_type='sigmoid'))
model.add(Dense(output_dim=1))
model.add(Activation(func_type='sigmoid', bias=False))

model.compose(loss = Loss('squared_error'))
err = model.fit(D[['x1','x2']].as_matrix(), D['y'].as_matrix(), epoch=1000, alpha = 0.01)

Y_pred = model.predict(D[['x1','x2']].as_matrix())
