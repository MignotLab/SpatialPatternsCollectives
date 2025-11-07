import numpy as np

def replace_error_msg(df,bact_index,error_msg, poles):
    if poles == [1]:
        df.loc[bact_index,"error_message_pole_1"] = error_msg
    elif poles == [2]:
        df.loc[bact_index,"error_message_pole_2"] = error_msg
    else:
        df.loc[bact_index,"error_message_pole_1"] = error_msg
        df.loc[bact_index,"error_message_pole_2"] = error_msg

def add_error_msg(df,bact_index,error_msg, poles):
    if poles == [1]:
        df.loc[bact_index,"error_message_pole_1"] = df.loc[bact_index,"error_message_pole_1"] + " | " + error_msg
    elif poles == [2]:
        df.loc[bact_index,"error_message_pole_2"] = df.loc[bact_index,"error_message_pole_2"] + " | " + error_msg
    else:
        df.loc[bact_index,"error_message_pole_1"] = df.loc[bact_index,"error_message_pole_1"] + " | " + error_msg
        df.loc[bact_index,"error_message_pole_2"] = df.loc[bact_index,"error_message_pole_2"] + " | " + error_msg

def replace_or_add(df, bact_index):
    # check if we have do add or replace an error message for pole 1 or 2
    if df.loc[bact_index,"error_message_pole_1"] == "no error":
        replace_pole_1 = True
    else:
        replace_pole_1 = False
    
    if df.loc[bact_index,"error_message_pole_2"] == "no error":
        replace_pole_2 = True
    else:
        replace_pole_2 = False
    
    return(replace_pole_1, replace_pole_2)

def write_error_msg(df,bact_index,error_msg, replace_pole_1, replace_pole_2, affected_poles):
    
    #write only to pole 1
    if affected_poles == [1]:
    # error msg pole 1
        if replace_pole_1 == True:
            replace_error_msg(df,bact_index,error_msg, poles = [1])
        else:
            add_error_msg(df,bact_index,error_msg, poles = [1])
    
    # write only to pole 2
    elif affected_poles == [2]:
        # error msg pole 2
        if replace_pole_2 == True:
            replace_error_msg(df,bact_index,error_msg, poles = [2])
        else:
            add_error_msg(df,bact_index,error_msg, poles = [2])

    # write to both poles
    else:
        if replace_pole_1 == True:
            replace_error_msg(df,bact_index,error_msg, poles = [1])
        else:
            add_error_msg(df,bact_index,error_msg, poles = [1])
        if replace_pole_2 == True:
            replace_error_msg(df,bact_index,error_msg, poles = [2])
        else:
            add_error_msg(df,bact_index,error_msg, poles = [2])
        

def create_error_msg_fct(df,bact_index,error_name,affected_poles):
    
    # initialize error message list
    error_msg_list = {
    "boundary": "Excluded: Bacteria too close to the boundary",
    "skan" : "Error: Problem with skan. Probably because only 0 or 1 poles were detected.",
    "0 or 1 poles": "Excluded: 0 or 1 poles detected",
    "3 or more poles": "Excluded: 3 or more poles detected",
    "skel too small": "Excluded: Skeleton below minimal length",
    "fluo thresh": "Error: Fluo threshold not hit anywhere --> summation over zero pixels",
    "small pole dif": "Warning: Pole difference generally too small",
    "noise dominates": "Warning: Pole difference small compared to tol_factor * noise mean standard deviation",
    "poles off": "Warning: No pole is definitely on",
    "poles on": "Warning: Both poles are on --> probably pollution",
    "no lead pole thresh hit": "Warning: No pole is above the leading pole threshold",
    "dead bacterium": "Excluded from reversal analysis: Bacteria is dead.",
    "traj short": "Excluded from reversal analysis: Trajectory too short."
    }

    # check at which pole we have to replace and at which we have to add an error msg
    replace_pole_1, replace_pole_2 = replace_or_add(df, bact_index)

    # write error message depending on error name 
    error_msg = error_msg_list[error_name]
    write_error_msg(df,bact_index,error_msg, replace_pole_1, replace_pole_2, affected_poles)



