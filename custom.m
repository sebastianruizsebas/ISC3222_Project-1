% Compare different noise levels
noise_levels = [0.01, 0.05, 0.1];
results = cell(1, 3);

for i = 1:3
    model = EulerModel(0.01, 10, 0.1, 1.0);
    model.generateSensoryInput(noise_levels(i));
    model.run();
    results{i} = model;
end

% Plot comparison
figure;
hold on;
for i = 1:3
    plot(results{i}.t, results{i}.free_energy, ...
         'DisplayName', sprintf('Noise = %.2f', noise_levels(i)));
end
legend; xlabel('Time (s)'); ylabel('Free Energy');
title('Effect of Sensory Noise on Free Energy');