def sliding_window(list_of_dataframe,window_size,forecast,time_step,emb_size):
    X, Y = [], [] #for stroing sliding windows and it results
    for i in range(0,len(list_of_dataframe)):
    
        data = list_of_dataframe[i][::-1]

        bpSys= data.loc[:, 'bpSys'].tolist()
        pulse = data.loc[:, 'pulse'].tolist()
        bpDia = data.loc[:, 'bpDia'].tolist()
        temp = data.loc[:, 'temp'].tolist()
        resp = data.loc[:, 'resp'].tolist()
        spo2 = data.loc[:, 'spo2'].tolist()

        score = data.loc[:, 'score_max'].tolist()

        WINDOW = window_size
        EMB_SIZE = emb_size
        STEP = time_step
        FORECAST = forecast
        
        #Do normalisation patient wise, though this also creates a lot of Nans too.
        
        
        
    
        for i in range(0, len(data)-15, STEP): 
            try:
                bpSys_window = bpSys[i:i+WINDOW]
                pulse_window = pulse[i:i+WINDOW]
                bpDia_window = bpDia[i:i+WINDOW]
                temp_window = temp[i:i+WINDOW]
                resp_window = resp[i:i+WINDOW]
                spo2_window = spo2[i:i+WINDOW]

                #I am normalising it in the sliding window,normalising it before can also be tried
                #standard scaler with fit transform can be used in place of this
                
                #Not recommended. A lot of Nan values creep in because there is a chance all values in the window are zero,and diving by zero makesit nan
                #This slows down the calculation a lot.
                # Using this right now. Will change later. replace nans with zeroes
                
                bpSys_window = (np.array(bpSys_window) - np.mean(bpSys_window)) / np.std(bpSys_window)
                pulse_window = (np.array(pulse_window) - np.mean(pulse_window)) / np.std(pulse_window)
                bpDia_window = (np.array(bpDia_window) - np.mean(bpDia_window)) / np.std(bpDia_window)
                temp_window = (np.array(temp_window) - np.mean(temp_window)) / np.std(temp_window)
                resp_window = (np.array(resp_window) - np.mean(resp_window)) / np.std(resp_window)
                spo2_window = (np.array(spo2_window) - np.mean(spo2_window)) / np.std(spo2_window)
                
                
                
                

                ews_scores = score[i:i+WINDOW]
                ews_score_three_ahead = score[i+WINDOW+FORECAST]
                ews_score_two_ahead = score[i+WINDOW+FORECAST-1]


                if int(ews_score_two_ahead)>2 and int(ews_score_three_ahead>2) :
                    y_i = [1]
                else:
                    y_i = [0] 
                
                #use column_stack instead of vstack https://stackoverflow.com/questions/16473042/numpy-vstack-vs-column-stack
                x_i = np.column_stack((bpSys_window, pulse_window, bpDia_window, temp_window, resp_window,spo2_window))

            except Exception as e:
                    print(e)
                    break
            X.append(x_i)
            Y.append(y_i)
            
    return X,Y
