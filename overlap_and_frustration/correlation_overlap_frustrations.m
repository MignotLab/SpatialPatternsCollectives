function correlation_overlap_frustrations(T)
v0=0.9564; %mean velocity of isolated bacteria in data set
K=T;
result_kstest=NaN(max(T.frame),1);
mean_frust_overlap_pertime=NaN(max(T.frame),1);
mean_frust_no_overlap_pertime=NaN(max(T.frame),1);
all_frust_overlap=[];
all_frust_no_overlap=[];
lineWidthFactor=200;
theta = linspace(0, 2 * pi, 100);
membraneWidthFactor = 50;
magnitude = v0;
b = magnitude;
a = b * membraneWidthFactor;
for time=0:max(T.frame)
    vap_x=[];
    vap_y=[];
    frust=[];

    kki=K(K.frame==time,:);
    for j=1:size(kki,1)
        [vx,vy]=compute_vap(kki,v0,j);
        vap_x=[vap_x; vx];
        vap_y=[vap_y; vy];
        frust=[frust; kki.local_frustration_s(j)];
    end
    x_centers_old=kki.x5(:);
    y_centers_old=kki.y5(:);

    x_centers_new= x_centers_old+vap_x;
    y_centers_new=y_centers_old+vap_y;
    scale=1.1;
    overlapAreas = cell(size(x_centers_new, 1), 1);
    overlapIndices = zeros(size(x_centers_new, 1), 1);

    for i = 1:size(x_centers_new, 1)
        u = vap_x(i);
        v = vap_y(i);

        u_unit = u / magnitude;
        v_unit = v / magnitude;

        xc = x_centers_new(i) + a * cos(theta) * u_unit - b * sin(theta) * v_unit;
        yc = y_centers_new(i) + a * cos(theta) * v_unit + b * sin(theta) * u_unit;

        overlapAreas{i} = [xc', yc'];
    end

    for i = 1:size(x_centers_new, 1)
        for j = i + 1:size(x_centers_new, 1)
            if any(inpolygon(overlapAreas{j}(:, 1), overlapAreas{j}(:, 2), overlapAreas{i}(:, 1), overlapAreas{i}(:, 2)))
                overlapIndices(i) = 1;
                overlapIndices(j) = 1;
            end
        end
    end
    frustColored = frust(overlapIndices == 1);
    frustwithout=frust;
    frustwithout(overlapIndices == 1) = [];
    nanIndices1 = find(isnan(frustColored));
    frustColored(nanIndices1)=[];
    nanIndices2 = find(isnan(frustwithout));
    frustwithout(nanIndices2)=[];
    mean_frust_overlap_pertime(time+1)=mean(frustColored(~isnan(frustColored)));
    mean_frust_no_overlap_pertime(time+1)=mean(frustwithout(~isnan(frustwithout)));
end


figure()
histogram(frustwithout,50,'Normalization','probability')
hold on
histogram(frustColored,50,'Normalization','probability')
xlabel('Frustration');
ylabel('Probability');

figure()
histogram(frust, 50, 'Normalization', 'probability');
xlabel('Frustration');
ylabel('Probability');

q = quiver(x_centers_new, y_centers_new, vap_x, vap_y, scale);
axis([-50 3400 -50 3400]);
q.LineWidth = 1;
xlabel('X-axis');
ylabel('Y-axis');

