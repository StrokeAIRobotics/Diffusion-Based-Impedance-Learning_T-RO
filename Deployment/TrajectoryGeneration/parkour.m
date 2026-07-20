%% To calculate the robot kinematics, the script uses Exp[licit]-MATLAB
% The code is publicly availble:
% https://explicit-robotics.github.io/exp_user/installation.html#explicit-matlab

% ------------------------- CLEAN-UP -------------------------
close all; clear; clc;

% ------------------------- SIMULATION SETUP -------------------------
robot = iiwa14('high');
robot.init();

% ------------------------- CONFIGURATIONS -------------------------
q_all = [ ...
    [5.59, 50.76, 0.04, -86.34, -3.49, 40.25, 92.61];  % q_ini
    [32.27, 78.85, 0.04, -65.39, -14.50, 44.79, 127.84];  % q_1
    [8.96, 75.83, 0.04, -74.07, -21.92, 35.50, 111.93];  % q_2
    [2.12, 75.72, 0.04, -75.32, -30.38, 34.86, 112.61];  % q_3
    [-3.93, 76.15, 0.04, -73.97, -31.26, 34.13, 107.27];  % q_4
    [-8.97, 73.31, 0.04, -83.25, -38.85, 27.56, 110.85];  % q_5
    [-3.65, 69.56, 0.04, -99.71, -57.86, 20.10, 137.12];  % q_6
    [-1.51, 68.21, 0.05, -101.76, -9.94, 11.49, 92.59];  % q_7
    [10.10, 70.39, 0.04, -114.47, 112.37, 26.41, -58.93];  % q_8
    [17.97, 70.20, 0.04, -108.29, 107.48, 24.44, -44.81];  % q_9
    [13.52, 70.67, 0.04, -101.85, 89.76, 23.89, -30.27];  % q_10
    [-16.24, 76.22, 26.76, -99.76, -11.37, 8.49, 46.93];  % q_11
    [5.59, 50.76, 0.04, -86.34, -3.49, 40.25, 92.61];  % q_final
]';
q_all = deg2rad(q_all);

% ------------------------- TIMING -------------------------
dt = 0.005;
T_all = [6.0, 9.5, 3.0, 2.5, 3.0, 2.8, 3.0, 3.5, 3.0, 2.5, 4.0, 6.0];
blend_time = 0.6;
mjt = @(p0, pf, tau) p0 + (pf - p0) .* (10*tau.^3 - 15*tau.^4 + 6*tau.^5);
blendfun = @(tau) 3*tau.^2 - 2*tau.^3;

% ------------------------- Z-POSITION REFERENCE -------------------------
H_q1 = robot.getForwardKinematics(q_all(:,2), 'bodyID', 7);
z_fixed = H_q1(3,4) - 0.015;

% ------------------------- INIT TRAJECTORY ARRAYS -------------------------
p_traj = [];
q_traj = [];

% ------------------------- FIRST SEGMENT (NO BLENDING): q_ini → shifted_q1 -------------------------
H0 = robot.getForwardKinematics(q_all(:,1), 'bodyID', 7);
R0 = H0(1:3,1:3); q0 = rotm2quat(R0)'; p0 = H0(1:3,4);

R1 = H_q1(1:3,1:3); q1 = rotm2quat(R1)';
p1 = H_q1(1:3,4); p1(3) = z_fixed;

T = T_all(1); t = 0:dt:T; n = length(t);
pos = zeros(3,n); quat = zeros(4,n);
for k = 1:n
    tau = t(k)/T;
    pos(:,k) = mjt(p0, p1, tau);
    quat(:,k) = quatinterp(q0', q1', blendfun(tau), 'slerp')';
end

p_traj = [p_traj, pos];
q_traj = [q_traj, quat];

% ------------------------- MID SEGMENTS WITH BLENDING: q1 → q2 → … → q11 -------------------------
blend_pts = round(blend_time / dt / 2);

for i = 2:11
    % From q_i to q_{i+1}
    H0 = robot.getForwardKinematics(q_all(:,i), 'bodyID', 7);
    H1 = robot.getForwardKinematics(q_all(:,i+1), 'bodyID', 7);

    p0 = H0(1:3,4); p0(3) = z_fixed;
    p1 = H1(1:3,4); p1(3) = z_fixed;
    q0 = rotm2quat(H0(1:3,1:3))';
    q1 = rotm2quat(H1(1:3,1:3))';

    T = T_all(i); t = 0:dt:T; n = length(t);
    pos = zeros(3,n); quat = zeros(4,n);
    for k = 1:n
        tau = t(k)/T;
        pos(1:2,k) = mjt(p0(1:2), p1(1:2), tau);
        pos(3,k) = z_fixed;
        quat(:,k) = quatinterp(q0', q1', blendfun(tau), 'slerp')';
    end

    % Apply blending: remove overlap and blend with previous
    if i == 2
        p_traj = [p_traj, pos(:,1:end-blend_pts)];
        q_traj = [q_traj, quat(:,1:end-blend_pts)];
    else
        % Blend last 'blend_pts*2' of old with first 'blend_pts*2' of new
        blend_range = 2*blend_pts;
        alpha = linspace(0,1,blend_range);
        alpha = blendfun(alpha);

        p_blend = (1 - alpha) .* p_traj(:,end-blend_range+1:end) + alpha .* pos(:,1:blend_range);
        q_blend = zeros(4, blend_range);
        for b = 1:blend_range
            q_blend(:,b) = quatinterp(q_traj(:,end-blend_range+1+b-1)', quat(:,b)', alpha(b), 'slerp')';
        end

    end
end

% ------------------------- FINAL SEGMENT (NO BLENDING): q11 → q_final -------------------------
H_last = robot.getForwardKinematics(q_all(:,12), 'bodyID', 7);
H_end = robot.getForwardKinematics(q_all(:,13), 'bodyID', 7);
p_last = H_last(1:3,4); 
p_last(3) = z_fixed; 
q_last = rotm2quat(H_last(1:3,1:3))';
p_end = H_end(1:3,4); 
q_end = rotm2quat(H_end(1:3,1:3))';

T = T_all(12); t = 0:dt:T; n = length(t);
pos = zeros(3,n); quat = zeros(4,n);
for k = 1:n
    tau = t(k)/T;
    pos(:,k) = mjt(p_last, p_end, tau);
    quat(:,k) = quatinterp(q_last', q_end', blendfun(tau), 'slerp')';
end

p_traj = [p_traj, pos(:,2:end)];
q_traj = [q_traj, quat(:,2:end)];

% ============================================================
% ------------------- FIX QUATERNION CONTINUITY --------------
% ============================================================
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
title('Full Position Trajectory'); xlabel('Time [s]'); ylabel('Position [m]');
legend('x','y','z'); grid on;

subplot(3,1,2);
plot(t_new, q_traj', 'LineWidth', 2);
title('Quaternion Components'); xlabel('Time [s]'); ylabel('Value');
legend('w','x','y','z'); grid on;

euler_angles = zeros(size(q_traj,2), 3);
for k = 1:size(q_traj,2)
    R = quat2rotm(q_traj(:,k)');
    euler_angles(k,:) = rad2deg(rotm2eul(R,'ZYX'));
end
euler_angles = unwrap(deg2rad(euler_angles))*180/pi;

subplot(3,1,3);
plot(t_new, euler_angles, 'LineWidth', 2);
title('Orientation (Euler Angles)'); xlabel('Time [s]'); ylabel('Angle [deg]');
legend('Yaw(Z)','Pitch(Y)','Roll(X)'); grid on;

% ------------------- FILE EXPORT -------------------
R_matrices = zeros(size(t_new,1),9);
for i = 1:size(t_new,1)
    R = quat2rotm(q_traj(:,i)');
    R_matrices(i,:) = reshape(R',1,9);
end

output_data = [t_new, p_traj', R_matrices];
output_file = '/YOUR_FOLDER/parkour_fullTrajectory.txt';
writematrix(output_data, output_file, 'Delimiter','tab');

fprintf('Full trajectory saved to:\n%s\n', output_file);