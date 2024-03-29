import numpy as np 
import tensorflow as tf
import os
import functions as func
import pandas as pd
import matplotlib.pyplot as plt
from time import time

#use sudo to overrap the error
#E tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1408] function cupti_interface_->Subscribe( &subscriber_, (CUpti_CallbackFunc)ApiCallback, this)failed with error CUPTI_ERROR_INSUFFICIENT_PRIVILEGES


os.environ['CUDA_VISIBLE_DEVICES']="1"
#The TF_ENABLE_XLA parameter, as defined, enables support for the Accelerated Linear Algebra (XLA) backend, make it super fast
#XLA (Accelerated Linear Algebra) is an open-source machine learning (ML) compiler for GPUs, CPUs, and ML accelerators.
#run export XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'


#control the GPU allocation
#config = tf.ConfigProto()
config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)
sess=tf.compat.v1.Session(config=config)

#1 frame used every 10 frame
frequency=1000


pdblist=func.get_all_pdbs()
#print(pdblist)

#number of pdbfile 
n_pdb=len(pdblist)

read_orignal_data=True

if read_orignal_data:
    all_bead_loc=[]
    all_bond_ang=[]
    all_trig_coord=[]
    all_amino_encode=[]
    all_close_amino_encode=[]
    all_close_amino_dist=[]
    all_invar_map=[]
    all_tri_pep_cg=[]
    all_tri_pep_aa=[]

    for pdb_id in pdblist:   
        
        #the direction saved precomputed data
        out_root="/home/yu/Desktop/mutiscale/outdata/"+pdb_id+"/"+pdb_id
        #load the data for each pdb_id
        all_bead_loct=func.loadnpy(out_root+"_all_bead_loc_"+str(frequency)+".npy")
        bls=np.shape(all_bead_loct)
        #print(all_bead_loct[1][0][1])
        all_bead_loct=np.reshape(all_bead_loct,(bls[0]*bls[1],bls[2],bls[3]))
        #print(np.shape(all_bead_loct))
        #print(all_bead_loct[10][1])
        
        all_bond_angt=func.loadnpy(out_root+"_all_bond_ang_"+str(frequency)+".npy")
        bas=np.shape(all_bond_angt)
        #print(all_bead_loct[1][0][1])
        all_bond_angt=np.reshape(all_bond_angt,(bas[0]*bas[1],bas[2],bas[3]))

        all_trig_coordt=func.loadnpy(out_root+"_all_trig_coord_"+str(frequency)+".npy")

        tcs=np.shape(all_trig_coordt)
        #print(all_bead_loct[1][0][1])
        all_trig_coordt=np.reshape(all_trig_coordt,(tcs[0]*tcs[1],tcs[2],tcs[3]))

        all_amino_encodet=func.loadnpy(out_root+"_all_amino_encode_"+str(frequency)+".npy")
        aes=np.shape(all_amino_encodet)
        #print(all_bead_loct[1][0][1])
        all_amino_encodet=np.reshape(all_amino_encodet,(aes[0]*aes[1],aes[2],aes[3]))
        #print(np.shape(all_amino_encodet))

        all_close_amino_encodet=func.loadnpy(out_root+"_all_close_amino_encode_"+str(frequency)+".npy")
        caes=np.shape(all_close_amino_encodet)
        #print(caes)
        #print(all_bead_loct[1][0][1])
        all_close_amino_encodet=np.reshape(all_close_amino_encodet,(caes[0]*caes[1],caes[2],caes[3],caes[4]))
        #print(np.shape(all_close_amino_encodet))
        #print(np.shape(all_close_amino_encodet))

        all_close_amino_distt=func.loadnpy(out_root+"_all_close_amino_dist_"+str(frequency)+".npy")
        cads=np.shape(all_close_amino_distt)
        #print(all_bead_loct[1][0][1])
        all_close_amino_distt=np.reshape(all_close_amino_distt,(cads[0]*cads[1],cads[2],cads[3]))
        #print(np.shape(all_close_amino_distt))

        all_amino_invar_mapt=func.loadnpy(out_root+"_all_invar_map_"+str(frequency)+".npy")
        aims=np.shape(all_amino_invar_mapt)
        #print(all_bead_loct[1][0][1])
        all_amino_invar_mapt=np.reshape(all_amino_invar_mapt,(aims[0]*aims[1],aims[2],aims[3],aims[4]))
        #print(np.shape(all_amino_invar_mapt))


        all_tri_pep_aat=func.loadnpy(out_root+"_all_tri_pep_aa_"+str(frequency)+".npy")
        tpas=np.shape(all_tri_pep_aat)
        #print(all_bead_loct[1][0][1])
        #combine protein and frame together
        all_tri_pep_aat=np.reshape(all_tri_pep_aat,(tpas[0]*tpas[1],tpas[2],tpas[3],tpas[4]))
        #print(np.shape(all_amino_invar_mapt))

        #append the data together

        all_bead_loc.append(all_bead_loct.astype(np.float32))
        all_bond_ang.append(all_bond_angt.astype(np.float32))
        all_trig_coord.append(all_trig_coordt.astype(np.float32))
        all_amino_encode.append(all_amino_encodet.astype(np.float32))
        all_close_amino_encode.append(all_close_amino_encodet.astype(np.float32))
        all_close_amino_dist.append(all_close_amino_distt.astype(np.float32))
        all_invar_map.append(all_amino_invar_mapt.astype(np.float32))
        all_tri_pep_aa.append(all_tri_pep_aat.astype(np.float32))

        print("append data together "+pdb_id)
        

    print("start concatenate")
    #print(np.shape(all_bead_loc))
    #comebine all data together 
    all_bead_loc=np.concatenate(all_bead_loc,axis=0)
    #    print(np.shape(all_bead_loc))
    #print("start concatenate")
    all_bond_ang=np.concatenate(all_bond_ang,axis=0)   
    #print(np.shape(all_bond_ang))
    #print("start concatenate")     
    all_trig_coord=np.concatenate(all_trig_coord,axis=0)
    #print(np.shape(all_trig_coord))
    #print("start concatenate")
    all_amino_encode=np.concatenate(all_amino_encode,axis=0)
    #print(np.shape(all_amino_encode))
    #print("start concatenate")
    all_close_amino_encode=np.concatenate(all_close_amino_encode,axis=0)
    #print(np.shape(all_close_amino_encode))
    #print("start concatenate")
    all_close_amino_dist=np.concatenate(all_close_amino_dist,axis=0)
    #print(np.shape(all_close_amino_dist))
    #print("start concatenate")
    all_invar_map=np.concatenate(all_invar_map,axis=0)
    #print(np.shape(all_invar_map))
    #print("start concatenate")
    
    # the output data
    all_tri_pep_aa=np.concatenate(all_tri_pep_aa,axis=0)
    #print(np.shape(all_tri_pep_aa))
    #print("end concatenate")

    #the number of triple, which also is the number of data
    ndat=all_bead_loc.shape[0]
    #print(ndat)

    #shuffle the dataset
    perm=np.random.permutation(np.arange(ndat))
    #print(perm)
    
    
    #shuffle the datas† by using the above index list
    #using the same index list, the data can still match each other
    all_bead_loc=all_bead_loc[perm]
    all_bond_ang=all_bond_ang[perm]
    all_trig_coord=all_trig_coord[perm]
    all_amino_encode=all_amino_encode[perm]
    all_close_amino_encode=all_close_amino_encode[perm]
    all_close_amino_dist=all_close_amino_dist[perm]
    all_invar_map=all_invar_map[perm]
    
    all_tri_pep_aa=all_tri_pep_aa[perm]

    out_perm="/home/yu/Desktop/mutiscale/outdata/"
    np.save(out_perm+"perm_",perm)


    

#####################################################################
#start bulid the model

print("specify the input")
#specify the input format of data
inp_specs=(tf.TensorSpec((3,3),dtype=tf.float32),#bead_loc
    tf.TensorSpec((3,3),dtype=tf.float32),#bond_ang
    tf.TensorSpec((3,3),dtype=tf.float32),#trig_coord
    tf.TensorSpec((3,20),dtype=tf.float32),#amino_encode
    tf.TensorSpec((3,15,20),dtype=tf.float32),#close_amino_encode
    tf.TensorSpec((3,15),dtype=tf.float32),#close_amino_dist
    tf.TensorSpec((3,27,26),dtype=tf.float32))#,#invar_map
    #tf.TensorSpec((3,26,25),dtype=tf.float32))#tri_pep_cg


#specify the out put format of datap
print("specify the output")
out_spec=tf.TensorSpec((3,26,3),dtype=tf.float32)


batch_size=256#256#256#1024
epochs=1
v_splt=0.3

#predefined parameters
#the number of nearest neighbor per atom
n_N=15
#the max number of atoms for each CG bead
#be careful about the n_M
n_M=27
#the number of CG bead per segment
n_B=3

dropout=0.001

#number of figerprin layers
num_fin_lay=3

#number of bead types
#L=hyper_dict["L"]#6#3#5
L=6

#init_encoding_w=hyper_dict["init_encoding_w"]#400
init_encoding_w=1
#40#width of first hidden layer for fingerprint
n_en_lay=3#number of layers for fingerprint

#np.linspace(start, stop, number of points)
fpt_ws=[int(x) for x in np.linspace(init_encoding_w,L,n_en_lay)]

lrate=0.00025


#f(x) = alpha * x if x < 0)
#f(x) = x if x >= 0
#modify the value a little bit if it smaller than 0
nl_act_G=tf.keras.layers.LeakyReLU(alpha=0.1)

#n_M is the number of max atoms per CG bead
out_dim=(n_M-1)*3
#generate a list of integers for the input layers, the start points may not be correct
gen_shapes=[int(x) for x in np.linspace(1,out_dim,4)]#last is n_tor_states 2500 was initial best

 #If no devices are specified in the constructor argument of the strategy then it will use all the available GPUs. If no GPUs are found, it will use the available CPUs. 
#Synchronous training across multiple replicas on one machine.

##used to train with mutiple GPU
#mirrored_strategy = tf.distribute.MirroredStrategy()
#with mirrored_strategy.scope():

if True:
    
    #initial the input data structure
    print("initial the input data structure")
    inp_bead_loc,inp_bond_ang,inp_trig_coord,inp_amino_encode,inp_close_amino_encode,inp_close_amino_dist,inp_invar_map=func.InitialInput(n_B=n_B,n_N=n_N,n_M=n_M)

    #build feed forward neural network

    #######Encode CG bead type#####################

    print("encode cg bead type")
    bead_i_out,bead_n_out=func.EncodeBeadType(n_en_lay=n_en_lay,fpt_ws=fpt_ws,nl_act_G=nl_act_G,inp_amino_encode=inp_amino_encode,inp_close_amino_encode=inp_close_amino_encode,dropout=dropout)

    ###############################################
    

    ###Calculate the symetry functions#############
    #shape([None, 3, 240])
    flat_sym_funs=func.radialSysmetryFunction(inp_close_amino_dist=inp_close_amino_dist,all_close_amino_dist=all_close_amino_dist,bead_i_out=bead_i_out,bead_n_out=bead_n_out,L=4,H=16,r_cut=15,n_B=3,plot=True)
    ###############################################


    ####flate bead_i_out array#####################
    #print("flat_sym_funs",flat_sym_funs.shape)
    #print(bead_i_out.shape)
    #reshpae
    bead_i_out_flat=tf.reshape(bead_i_out,(tf.shape(bead_i_out)[0],bead_i_out.shape[1]*bead_i_out.shape[2]))
    #print(bead_i_out_flat.shape)
    #number of bead 3 arrays 
    #shape[None,3,18]
    bead_i_out_flastack=tf.stack([bead_i_out_flat for ia in range(n_B)],axis=1)
    #print("bead_i_out_flastack",bead_i_out_flastack.shape)
    ################################################

    #concate tensors(a few input data together) along one direction
    inp_dnn=tf.concat([bead_i_out_flastack,inp_bead_loc,inp_amino_encode,inp_bond_ang,flat_sym_funs],axis=2)
    #print("inp_dnn",inp_dnn.shape)


    gen_reg=tf.keras.regularizers.L2(l2=0.25)#025)

    ###########bulid layers###############
    den_1=tf.keras.layers.Dense(gen_shapes[0],activation=nl_act_G,kernel_regularizer=gen_reg,use_bias=True)
    dro_1=tf.keras.layers.Dropout(dropout)
    out_1=dro_1(den_1(inp_dnn))


    den_2=tf.keras.layers.Dense(gen_shapes[1],activation=nl_act_G,kernel_regularizer=gen_reg,use_bias=True)
    dro_2=tf.keras.layers.Dropout(dropout)
    out_2=dro_2(den_2(out_1))


    den_3=tf.keras.layers.Dense(gen_shapes[2],activation=nl_act_G,kernel_regularizer=gen_reg,use_bias=True)
    dro_3=tf.keras.layers.Dropout(dropout)
    ##combine residual
    #out_3=dro_3(den_3(tf.concat([out_2,inp_dnn],axis=-1)))
    out_3=dro_3(den_3(out_2))


    den_4=tf.keras.layers.Dense(gen_shapes[3],activation=nl_act_G,use_bias=True)
    dnn_output=den_4(out_3)#out_1)
    dnn_output=tf.reshape(dnn_output,[tf.shape(dnn_output)[0],3,n_M-1,3])
    
    #apply CG invar mapping
    aa_res_frame=tf.matmul(inp_invar_map,dnn_output)
    print(aa_res_frame.shape)
    #apply tripeptide internal translations
    #return the ddition of two tensors
    #tf.expand_dims: Returns a tensor with a length 1 axis inserted at index axis.
    aa_trip_frame=tf.add(aa_res_frame,tf.expand_dims(inp_trig_coord,axis=2))
    print(aa_trip_frame.shape)
    
    #now RMSD fit to the true coordinates...
    #tack on the pred 'cg' positions
    #expand one dimension
    #aa_trip_frame=tf.concat([aa_trip_frame,tf.expand_dims(inp_trig_coord,axis=2)],axis=2)
    print(aa_trip_frame.shape)

    ######################################
    print("start model")
    model = tf.keras.Model(inputs=[inp_bead_loc
                               ,inp_bond_ang
                               ,inp_trig_coord
                               ,inp_amino_encode
                               ,inp_close_amino_encode
                               ,inp_close_amino_dist
                               ,inp_invar_map],outputs=aa_trip_frame)


opt = tf.keras.optimizers.Adam(learning_rate=lrate)#, beta_1=0.5)
model.compile(loss=func.align_loss, optimizer=opt, metrics=[])

    #### for scope ^^
#model.summary()

#director save log data

out_dir="/home/yu/Desktop/mutiscale/log/test/"    
log_dir = out_dir+'protdet_log' 

#turn of warning,somehow it works
#option=tf.data.Options()
#option.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
#all_data=tf.data.Dataset.from_tensor_slices((all_bead_loc,all_tri_pep_aa))
################

print("start call back")
#Enable visualizations for TensorBoard.
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,write_graph=True,update_freq='epoch')

print(np.shape(all_tri_pep_aa))


print("start fit")

print(np.shape(all_bead_loc))
print(np.shape(all_bond_ang))
print(np.shape(all_trig_coord))
print(np.shape(all_amino_encode))
print(np.shape(all_close_amino_encode))
print(np.shape(all_invar_map))
print(np.shape(all_tri_pep_aa))



#first 10 values
"""
model.fit([all_bead_loc[0:10],
            all_bond_ang[0:10],
            all_trig_coord[0:10],
            all_amino_encode[0:10],
            all_close_amino_encode[0:10],
            all_close_amino_dist[0:10]
            ,all_invar_map[0:10]],all_tri_pep_aa[0:10],batch_size=batch_size,epochs=epochs,validation_split=v_splt,validation_batch_size=batch_size)
"""

#all values
history=model.fit([all_bead_loc,
            all_bond_ang,
            all_trig_coord,
            all_amino_encode,
            all_close_amino_encode,
            all_close_amino_dist
            ,all_invar_map],all_tri_pep_aa,batch_size=batch_size,epochs=epochs,validation_split=v_splt,validation_batch_size=batch_size)


print("start save")
model.save(out_dir+'protdet.tf')

print("all_tri_pep_aa")
print(all_tri_pep_aa[0:1])
print("all_trig_coord")
print(all_trig_coord[:1])

test=model.predict([all_bead_loc[0:1],
            all_bond_ang[0:1],
            all_trig_coord[0:1],
            all_amino_encode[0:1],
            all_close_amino_encode[0:1],
            all_close_amino_dist[0:1]
            ,all_invar_map[0:1]])

print(test)
print(np.shape(all_tri_pep_aa[0:1]),np.shape(test))


#####
testd="../test.xyz"
refd="../ref.xyz"
cosd="../cos.xyz"
func.savetoXYZ(test,testd)
func.savetoXYZ(all_tri_pep_aa[0:1],refd)
print(all_trig_coord.shape)
print("start")
func.saveCGtoXYZ(all_trig_coord[0:1],cosd)



##########################
plt.figure()
print("model.history")
print(model.history.history)
print(history.history.keys())
print(history.epoch)
plt.plot(history.epoch, history.history["loss"], 'g', label='Training loss')
plt.plot(history.epoch, history.history["val_loss"], 'b', label='Val loss')
plt.title('loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss.png")
plt.show()
#########################
#pd.DataFrame(model.history.history).plot(figsize=(8,5))
#plt.savefig("loss.png")
#plt.show()

#pltmod=1
#val_bead_loc=all_bead_loc[int((1-v_splt)*all_bead_loc.shape[0])::pltmod]
#val_bond_ang=all_bond_ang[int((1-v_splt)*all_bead_loc.shape[0])::pltmod]
#val_trig_coord=all_trig_coord[int((1-v_splt)*all_bead_loc.shape[0])::pltmod]
#val_amino_encode=all_amino_encode[int((1-v_splt)*all_bead_loc.shape[0])::pltmod]
#val_close_amino_encode=all_close_amino_encode[int((1-v_splt)*all_bead_loc.shape[0])::pltmod]
#val_close_amino_dist=all_close_amino_dist[int((1-v_splt)*all_bead_loc.shape[0])::pltmod]
#val_invar_map=all_invar_map[int((1-v_splt)*all_bead_loc.shape[0])::pltmod]
#val_tri_pep_aa=all_tri_pep_aa[int((1-v_splt)*all_bead_loc.shape[0])::pltmod]
#val_pred=model.predict([val_bead_loc
#                       ,val_bond_ang
#                       ,val_trig_coord
#                       ,val_amino_encode
#                       ,val_close_amino_encode
#                       ,val_close_amino_dist
#                       ,val_invar_map],batch_size=int(bsize))
#
#align_trans=align_pred(val_tri_pep_aa, val_pred)
#align_trans=np.array(align_trans)
#val_tri_pep_aa=np.array(val_tri_pep_aa)[:,:,:n_M]
#
#plt.hist2d(val_tri_pep_aa[val_tri_pep_aa!=0.].flatten(),(align_trans[val_tri_pep_aa!=0.].flatten()-val_tri_pep_aa[val_tri_pep_aa!=0.].flatten()),bins=100)#,alpha=0.01)
#plt.xlabel("True position / Angstroms")
#plt.ylabel("Pred - True position / Angstroms")
#plt.show()
