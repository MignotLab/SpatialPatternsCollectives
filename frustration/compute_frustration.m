function [cumul_frust, vr_mod]=compute_frustration(TP,kki,d,eloi,frustration_formula,c,v0)

vr_x=[]; vr_y=[]; vap_x=[];vap_y=[];
sumexpo=sum(exp(-(0:d-1)/d));
number_of_frames=1;
for k= TP-eloi:-1:TP-d-eloi+1
    c=c+1;
    vap_x(c)=kki.x_main_pole(k)-kki.x5(k);
    vap_y(c)=kki.y_main_pole(k)-kki.y5(k);
    dir=sqrt(vap_y(c)^2+vap_x(c)^2);
    vap_x(c)=v0*vap_x(c)/dir;
    vap_y(c)=v0*vap_y(c)/dir;

    vr_x(c)=kki.x_main_pole(k+1)-kki.x_main_pole(k)/number_of_frames;
    vr_y(c)=kki.y_main_pole(k+1)-kki.y_main_pole(k)/number_of_frames;

    switch frustration_formula
        case 'ps_max'
            frust(c)=(1-(vap_x(c)*vr_x(c)+vap_y(c)*vr_y(c))/max(vap_x(c)^2+vap_y(c)^2,vr_y(c)^2+vr_x(c)^2))*exp(-(c-1)/d)/sumexpo;
        case'ps_max_sum'
            frust(c)=(1-(vap_x(c)*vr_x(c)+vap_y(c)*vr_y(c))/max(vap_x(c)^2+vap_y(c)^2,vr_y(c)^2+vr_x(c)^2));
        case 'mean'
            frust(c)=(1-(vap_x(c)*vr_x(c)+vap_y(c)*vr_y(c))/max(vap_x(c)^2+vap_y(c)^2,vr_y(c)^2+vr_x(c)^2));
    end
end
cumul_frust=sum(frust);
vr_mod(:)=sqrt(vr_y(:).^2+vr_x(:).^2);
