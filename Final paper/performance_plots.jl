using Plots

learning_rate_1 = [0.001, 0.01, 0.1]
time_1 = [500.0, 52.008, 20.632]

learning_rate_10 = [0.001, 0.01, 0.1]
time_10 = [350.039, 58.271, 48.579]

learning_rate_100 = [0.001, 0.01, 0.1]
time_100 = [285.687, 90.685, 89.037]


tolerance_1 = [6.0, 7.0, 8.0, 9.0 ]
time_1t = [20.632, 67.442, 101.719, 78.553]

tolerance_10 = [6.0, 7.0, 8.0, 9.0 ]
time_10t = [48.579, 32.137, 100.993, 402.402]

tolerance_100 = [6.0, 7.0, 8.0, 9.0 ]
time_100t = [89.037,500.0, 500.0, 500.0]





plot(learning_rate_1, time_1, title = "Time vs Learning Rate for Epochs 1, 10, 100", xlabel = "Learning rate", ylabel = "Time (s)", label = "Epoch 1", ylims = (0.0, 450.0))
plot!(learning_rate_10, time_10, label = "Epoch 10")
plot!(learning_rate_100, time_100, label = "Epoch 100")


plot(tolerance_1, time_1t, title = "Time vs Sovler Tolerance for Epochs 1, 10, 100", xlabel = "Solver Tolerance", ylabel = "Time (s)", label = "Epoch 1", ylims = (0.0, 450))
plot!(tolerance_10, time_10t, label = "Epoch 10")
plot!(tolerance_100, time_100t, label = "Epoch 100")
