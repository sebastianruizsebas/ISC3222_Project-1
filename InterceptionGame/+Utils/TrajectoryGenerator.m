classdef TrajectoryGenerator
    % TrajectoryGenerator - Generate target motion trajectories
    %
    % Provides static methods for creating various target motion patterns
    % Follows open science: reproducible with fixed random seeds
    %
    % Usage:
    %   [traj, times, v, a] = Utils.TrajectoryGenerator.generateTargetTrajectory(...);
    
    methods (Static)
        function [target_traj, target_times, v_true, a_true] = ...
                generateTargetTrajectory(width, height, speed_range, accel_type, duration)
            % Generate target trajectory
            %
            % Args:
            %   width, height: screen dimensions
            %   speed_range: [min_speed, max_speed] in pixels/sec
            %   accel_type: 'constant', 'accelerating', or 'decelerating'
            %   duration: trial duration in seconds
            %
            % Returns:
            %   target_traj: [n_frames, 2] matrix of x,y positions
            %   target_times: [n_frames, 1] time points
            %   v_true: true velocity
            %   a_true: true acceleration
            
            dt = 0.016;  % ~60 FPS
            target_times = (0:dt:duration)';
            
            % Generate trajectory based on acceleration type
            x_start = width * 0.2;
            x_end = width * 0.8;
            
            switch accel_type
                case 'constant'
                    v_true = speed_range(1) + rand()*(speed_range(2)-speed_range(1));
                    a_true = 0;
                    x_pos = x_start + v_true * target_times;
                    
                case 'accelerating'
                    v_true = speed_range(1) + rand()*50;
                    a_true = 50 + 50*rand();
                    x_pos = x_start + v_true * target_times + 0.5*a_true*target_times.^2;
                    
                case 'decelerating'
                    v_true = speed_range(2) - rand()*50;
                    a_true = -60 - 30*rand();
                    x_pos = x_start + v_true * target_times + 0.5*a_true*target_times.^2;
                    
                otherwise
                    error('Unknown acceleration type: %s', accel_type);
            end
            
            % Clip to screen bounds
            x_pos = max(50, min(width-50, x_pos));
            
            % Y position: sinusoidal motion
            y_pos = height/2 + 50*sin(2*pi*target_times/duration);
            
            target_traj = [x_pos, y_pos];
        end
        
        function [traj, times] = generateComplexTrajectory(width, height, duration)
            % Generate complex trajectory with direction change
            
            dt = 0.016;
            times = (0:dt:duration)';
            
            % Split into two phases
            t_split = duration/2;
            phase1_idx = times <= t_split;
            phase2_idx = times > t_split;
            
            % Phase 1: move right
            x_p1 = 100 + 150 * times(phase1_idx);
            y_p1 = height/2 * ones(sum(phase1_idx), 1);
            
            % Phase 2: move left
            times_p2 = times(phase2_idx) - t_split;
            x_start_p2 = x_p1(end);
            x_p2 = x_start_p2 - 100 * times_p2;
            y_p2 = height/2 * ones(sum(phase2_idx), 1);
            
            % Combine phases
            x_pos = [x_p1; x_p2];
            y_pos = [y_p1; y_p2];
            
            traj = [x_pos, y_pos];
        end
    end
end
