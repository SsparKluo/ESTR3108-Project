import global_cnn
import multi_channel_cnn
import predict

#name_list = ["CLIPSEQ_ELAVL1", "CLIPSEQ_SFRS1", "ICLIP_HNRNPC", "ICLIP_TDP43", "ICLIP_TIA1", "ICLIP_TIAL1", "PARCLIP_AGO1234", "PARCLIP_ELAVL1", "PARCLIP_ELAVL1A", "PARCLIP_EWSR1", "PARCLIP_FUS", "PARCLIP_HUR", "PARCLIP_IGF2BP123", "PARCLIP_MOV10_Sievers", "PARCLIP_PUM2", "PARCLIP_QKI", "PARCLIP_TAF15", "PTBv1", "ZC3H7B_Baltz2012"]
name_list = ["CLIPSEQ_ELAVL1"]

for name in name_list:
	predict.predict(name)