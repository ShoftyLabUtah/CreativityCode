# ''' PATIENT PARAMETERS FOR DMN LFP ANALYSIS'''

#### BLACKROCK RECORDINGS

data_loc1 = '/Volumes/155.100.91.44/Data/UIC202307/DMN/20230529-115352'
patient1 = 'UIC202307'
data_loc2 = '/Volumes/155.100.91.44/Data/UIC202311/DMN/20230905-085636'
patient2 = 'UIC202311'
data_loc3 = '/Volumes/155.100.91.44/Data/UIC202314/DMN/20231208-133604'
patient3 = 'UIC202314'
data_loc4 = '/Volumes/155.100.91.44/Data/UIC202308/DMN/20230802-122445' ####weird beh data?
patient4 = 'UIC202308'
# data_loc5a = '/Volumes/155.100.91.44/Data/UIC202401/DMN/20240111-121651' ###USE CHANNELMAP1
# patient5a = 'UIC202401'
data_loc5 = '/Volumes/155.100.91.44/Data/UIC202401/DMN/20240115-101116' ###USE CHANNELMAP2
patient5 = 'UIC202401'
data_loc6 = '/Volumes/155.100.91.44/Data/UIC202404/DMN/20240306-134703' 
patient6 = 'UIC202404'
data_loc7 = '/Volumes/155.100.91.44/Data/UIC202407/DMN/20240412-141822' 
patient7 = 'UIC202407'
data_loc8 = '/Volumes/155.100.91.44/Data/UIC202409/DMN/20240510-142059' 
patient8 = 'UIC202409'
data_loc9 = '/Volumes/155.100.91.44/Data/UIC202413/DMN/20240718-100231' 
patient9 = 'UIC202413'
data_loc10 = '/Volumes/155.100.91.44/Data/UIC202414/DMN/20240802-110702' 
patient10 = 'UIC202414'
data_loc11 = '/Volumes/155.100.91.44/Data/UIC202422/DMN/20250103-115359' 
patient11 = 'UIC202422'
data_loc12 = '/Volumes/155.100.91.44/Data/UIC202504/DMN/20250217-123943' 
patient12 = 'UIC202504' #use channelmap2 for macros; channelmap1 for micros
data_loc13 = '/Volumes/155.100.91.44/Data/UIC202505/DMN/20250227-120201' 
patient13 = 'UIC202505'
data_loc14 = '/Volumes/155.100.91.44/Data/UIC202521/DMN/UIC202501/20251213-150227' #use for OCT analysis
patient14 = 'UIC202521'
data_loc15 = '/Volumes/155.100.91.44/Data/UIC202522/DMN/UIC202522/20260101-125532' #use for OCT analysis
patient15 = 'UIC202522'


patients_data = [(patient1,data_loc1), (patient2,data_loc2), (patient3,data_loc3), (patient4,data_loc4), (patient5,data_loc5),
    (patient6,data_loc6), (patient7,data_loc7), (patient8,data_loc8), (patient9,data_loc9), (patient10,data_loc10), (patient11,data_loc11),
    (patient12,data_loc12), (patient13,data_loc13), (patient14,data_loc14), (patient15,data_loc15)]


# '''RIPPLE RECORDINGS'''

ripple_data_loc1 = '/Volumes/155.100.91.44/Data/UIC202418/SEEG/DMN/20241017-133031' 
ripple_patient1 = 'UIC202418' 

ripple_data_loc2 = '/Volumes/155.100.91.44/Data/UIC202420/DMN/20241114-124503' #bad dataset
ripple_patient2 = 'UIC202420' 

ripple_data_loc3 = '/Volumes/155.100.91.44/Data/UIC202421/DMN/20241206-124436'  
ripple_patient3 = 'UIC202421' 

ripple_data_loc4 = '/Volumes/155.100.91.44/Data/UIC202422/DMN/20241231-144952'  
ripple_patient4 = 'UIC202422' 


patients_data_ripple = [(ripple_patient1,ripple_data_loc1), (ripple_patient2,ripple_data_loc2), (ripple_patient3,ripple_data_loc3), (ripple_patient4,ripple_data_loc4)]


##### SINGLE UNIT RECORDINGS


spikes_filename1 = '/Volumes/155.100.91.44/Data/UIC202308/DMN/20230802-122445/20230802-122445-001_sortedRC.nev' 
spikes_data_loc1 = '/Volumes/155.100.91.44/Data/UIC202308/DMN/20230802-122445'
spikes_patient1= 'UIC202308'

spikes_filename2 = '/Volumes/155.100.91.44/Data/UIC202311/DMN/20230905-085636/20230905-085636-001_sortedRC.nev' 
spikes_data_loc2 = '/Volumes/155.100.91.44/Data/UIC202311/DMN/20230905-085636'
spikes_patient2 = 'UIC202311'

spikes_filename3 = '/Volumes/155.100.91.44/Data/UIC202314/DMN/20231208-133604/20231208-133604-_sortedRC.nev' 
spikes_data_loc3 = '/Volumes/155.100.91.44/Data/UIC202314/DMN/20231208-133604'
spikes_patient3 = 'UIC202314'

spikes_filename4 = '/Volumes/155.100.91.44/Data/UIC202404/DMN/20240306-134703/20240306-134703-_sortedRC.nev' 
spikes_data_loc4 = '/Volumes/155.100.91.44/Data/UIC202404/DMN/20240306-134703'
spikes_patient4 = 'UIC202404'

spikes_filename5 = '/Volumes/155.100.91.44/Data/UIC202407/DMN/20240412-141822/20240412-141822-sortedRC.nev' 
spikes_data_loc5 = '/Volumes/155.100.91.44/Data/UIC202407/DMN/20240412-141822'
spikes_patient5 = 'UIC202407'

spikes_filename6 = '/Volumes/155.100.91.44/Data/UIC202409/DMN/20240510-142059/20240510-142059-_sortedRC.nev' 
spikes_data_loc6 = '/Volumes/155.100.91.44/Data/UIC202409/DMN/20240510-142059'
spikes_patient6 = 'UIC202409'

spikes_filename7 = '/Volumes/155.100.91.44/Data/UIC202413/DMN/20240718-100231/20240718-100231-_sortedRC.nev' 
spikes_data_loc7 = '/Volumes/155.100.91.44/Data/UIC202413/DMN/20240718-100231'
spikes_patient7 = 'UIC202413'

spikes_filename8 = '/Volumes/155.100.91.44/Data/UIC202422/DMN/20250103-115359/20250103-115359-_sortedRC.nev' 
spikes_data_loc8 = '/Volumes/155.100.91.44/Data/UIC202422/DMN/20250103-115359'
spikes_patient8 = 'UIC202422'

spikes_filename9 = '/Volumes/155.100.91.44/Data/UIC202504/DMN/20250217-123943/20250217-123943-_sortedRC.nev' 
spikes_data_loc9 = '/Volumes/155.100.91.44/Data/UIC202504/DMN/20250217-123943'
spikes_patient9 = 'UIC202504'

spikes_filename10 = '/Volumes/155.100.91.44/Data/UIC202505/DMN/20250227-120201/20250227-120201-_sortedRC.nev' 
spikes_data_loc10 = '/Volumes/155.100.91.44/Data/UIC202505/DMN/20250227-120201' 
spikes_patient10 = 'UIC202505'

spikes_filename11 = '/Volumes/155.100.91.44/Data/UIC202507/DMN/20250402-124022/20250402-124022--001.nev'
spikes_data_loc11 = '/Volumes/155.100.91.44/Data/UIC202507/DMN/20250402-124022' 
spikes_patient11 = 'UIC202507'


patients_data_spikes = [(spikes_patient1, spikes_data_loc1, spikes_filename1), (spikes_patient2, spikes_data_loc2, spikes_filename2), 
                (spikes_patient3, spikes_data_loc3, spikes_filename3), (spikes_patient4, spikes_data_loc4, spikes_filename4), 
                (spikes_patient5, spikes_data_loc5, spikes_filename5), (spikes_patient6, spikes_data_loc6, spikes_filename6), 
                (spikes_patient7, spikes_data_loc7, spikes_filename7), (spikes_patient8, spikes_data_loc8, spikes_filename8), 
                (spikes_patient9, spikes_data_loc9, spikes_filename9), (spikes_patient10, spikes_data_loc10, spikes_filename10)]

