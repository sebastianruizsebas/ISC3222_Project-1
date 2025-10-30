function plot_trajectory_3d_dashed(x, y, z, base_color, alpha)
    % Plot a 3D trajectory with dashed line and fading effect
    %
    % Usage:
    %   plot_trajectory_3d_dashed(x, y, z, 'r', 0.7)
    %
    % Inputs:
    %   x, y, z: trajectory coordinates (vectors)
    %   base_color: color name ('r', 'g', 'b', 'm', etc.)
    %   alpha: transparency/fade parameter (currently unused, for future)
    
    n_points = length(x);
    if n_points < 2, return; end
    
    % Create color gradient from bright to dim
    color_rgb = hex2rgb(base_color);
    
    % Plot segments with decreasing brightness
    for i = 1:n_points-1
        progress = i / n_points;  % 0 to 1 over trajectory
        
        % Fade effect: bright at start, dim at end
        brightness = 1 - (progress * 0.4);  % Goes from 1 to 0.6
        segment_color = color_rgb * brightness;
        
        % Line width decreases slightly over time
        lw = 2.0 * (1 - progress * 0.3);
        
        plot3([x(i), x(i+1)], [y(i), y(i+1)], [z(i), z(i+1)], ...
            'Color', segment_color, 'LineWidth', lw, 'LineStyle', '--', 'HandleVisibility', 'off');
    end
end

function rgb = hex2rgb(color)
    % Convert color name to RGB
    switch color
        case 'r'
            rgb = [1, 0, 0];
        case 'g'
            rgb = [0, 1, 0];
        case 'b'
            rgb = [0, 0, 1];
        case 'm'
            rgb = [1, 0, 1];
        case 'c'
            rgb = [0, 1, 1];
        case 'y'
            rgb = [1, 1, 0];
        case 'k'
            rgb = [0, 0, 0];
        otherwise
            rgb = [0.5, 0.5, 0.5];
    end
end
