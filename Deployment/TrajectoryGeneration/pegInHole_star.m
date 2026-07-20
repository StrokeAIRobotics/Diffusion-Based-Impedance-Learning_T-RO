%% To calculate the robot kinematics, the script uses Exp[licit]-MATLAB
% The code is publicly availble:
% https://explicit-robotics.github.io/exp_user/installation.html#explicit-matlab

% ------------------------- CLEAN-UP -------------------------
close all; clear; clc;

% ------------------------- SIMULATION SETUP -------------------------
robot = iiwa14('high');
robot.init();

tool_offset = [0; 0; 0.05];  % 5 cm along tool z-axis
dt = 0.005;                  % Common time step
mjt = @(p0, pf, tau) p0 + (pf - p0) .* (10*tau.^3 - 15*tau.^4 + 6*tau.^5); % Minimum Jerk helper

% ------------------------- INITIAL POSE -------------------------
q_ini = deg2rad([18.06, 45.78, -0.02, -78.91, 2.08, 58.55, 40.97]');
H_ini = robot.getForwardKinematics(q_ini, 'bodyID', 7); 
p_ini = H_ini(1:3,4);
q_ini_quat = rotm2quat(H_ini(1:3,1:3))';

% Initialize trajectory arrays
p_traj = [];
q_traj = [];

% ------------------- FIRST MJT: ini → touch -------------------
q_first = deg2rad([5.28, 75.62, 15.14, -69.45, -2.65, 44.89, 41.72]');
H_first = robot.getForwardKinematics(q_first, 'bodyID', 7, 'position', -tool_offset);
p_first = H_first(1:3,4);
p_first(1) = p_first(1) - 0.002;  % 2 mm along -x
p_first(3) = p_first(3) - 0.003;  % 3 mm downward
q_first_quat = rotm2quat(H_first(1:3,1:3))';

T_first = 4.0; t_first = 0:dt:T_first;
pos_first = zeros(3, length(t_first));
quat_first = zeros(4, length(t_first));
for k = 1:length(t_first)
    tau = t_first(k)/T_first;
    pos_first(:,k) = mjt(p_ini, p_first, tau);
    quat_first(:,k) = quatinterp(q_ini_quat', q_first_quat', 10*tau^3-15*tau^4+6*tau^5, 'slerp')';
end
p_traj = [p_traj, pos_first];
q_traj = [q_traj, quat_first];

% ------------------- SECOND MJT (UPDATED): touch → final -------------------
q_fin = deg2rad([-2.09, 83.60, 34.94, -70.46, -51.95, 44.93, 71.07]');
H_fin = robot.getForwardKinematics(q_fin, 'bodyID', 7, 'position', -tool_offset);
p_fin = H_fin(1:3,4);
p_fin(3) = p_fin(3) - 0.035; % 35 mm downward
q_fin_quat = rotm2quat(H_fin(1:3,1:3))';

T_fin = 6.0; t_fin = 0:dt:T_fin;
p_start_fin = p_traj(:,end); q_start_fin = q_traj(:,end);
pos_fin = zeros(3, length(t_fin));
quat_fin = zeros(4, length(t_fin));
for k = 1:length(t_fin)
    tau = t_fin(k)/T_fin;
    pos_fin(:,k) = mjt(p_start_fin, p_fin, tau);
    quat_fin(:,k) = quatinterp(q_start_fin', q_fin_quat', 10*tau^3-15*tau^4+6*tau^5, 'slerp')';
end
p_traj = [p_traj, pos_fin(:,2:end)];
q_traj = [q_traj, quat_fin(:,2:end)];

% ------------------- THIRD MJT: final → up -------------------
p_start_up = p_traj(:,end);
p_goal_up = [p_start_up(1);
             p_start_up(2);
             p_start_up(3) + 0.30]; % 30 cm upward
T_up = 6.0; t_up = 0:dt:T_up;
pos_up = zeros(3, length(t_up));
quat_up = repmat(q_traj(:,end),1,length(t_up));
for k = 1:length(t_up)
    tau = t_up(k)/T_up;
    pos_up(:,k) = mjt(p_start_up, p_goal_up, tau);
end
p_traj = [p_traj, pos_up(:,2:end)];
q_traj = [q_traj, quat_up(:,2:end)];

% ------------------- FIX QUATERNION CONTINUITY -------------------
for i = 2:size(q_traj,2)
    if dot(q_traj(:,i), q_traj(:,i-1)) < 0
        q_traj(:,i) = -q_traj(:,i);
    end
end

% ------------------- TIME VECTOR -------------------
t_new = (0:dt:(size(p_traj,2)-1)*dt)';

% ------------------- PLOTS -------------------
figure;
subplot(3,1,1);
plot(t_new, p_traj', 'LineWidth', 2);
title('Full Position Trajectory');
xlabel('Time [s]'); ylabel('Position [m]');
legend('x','y','z'); grid on;

subplot(3,1,2);
plot(t_new, q_traj', 'LineWidth', 2);
title('Quaternion Components');
xlabel('Time [s]'); ylabel('Value');
legend('w','x','y','z'); grid on;

% Euler Angles
euler_angles = zeros(size(q_traj,2),3);
for k = 1:size(q_traj,2)
    R = quat2rotm(q_traj(:,k)');
    euler_angles(k,:) = rad2deg(rotm2eul(R,'ZYX'));
end
euler_angles = unwrap(deg2rad(euler_angles))*180/pi;
subplot(3,1,3);
plot(t_new, euler_angles, 'LineWidth', 2);
title('Orientation (Euler Angles)');
xlabel('Time [s]'); ylabel('Angle [deg]');
legend('Yaw(Z)','Pitch(Y)','Roll(X)'); grid on;

% ------------------- FILE EXPORT -------------------
R_matrices = zeros(size(t_new,1),9);
for i = 1:size(t_new,1)
    R = quat2rotm(q_traj(:,i)');
    R_matrices(i,:) = reshape(R',1,9);
end
output_data = [t_new, p_traj', R_matrices];
output_file = '/YOUR_FOLDER/pegInHole_star_fullTrajectory.txt';
writematrix(output_data, output_file, 'Delimiter','tab');
fprintf('Full trajectory (ini → first → final → up) saved to:\n%s\n', output_file);