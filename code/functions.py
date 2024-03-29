import numpy as np
import MDAnalysis as mda
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit,prange
from time import time
import subprocess as sp
import multiprocessing as mp
import os
import sys
import os
import tensorflow as tf

#CG cartersion positions 3 bead segment
#CG cartersion positions N-closest neighbors 
#Bead identitity(residue) 3 bead segment
#Bead identity (residue) N-closest near neighbor
#pair distance 3 bead distance
#CG 2 bead 1 angle
#tripeptide frame coordinate
#CG map psedudoinverse


# a simple comment to read the position from MDAnalysis Universe
# unvi: all data object of mpd
#frenquency: get 1 frame per frequency frame
#posit: 

def get_posits(univ,frequency):
    select=univ.select_atoms("all")
    posit=[]
    boxes=[]
    #[start:end:frequency]
    for ts in univ.trajectory[::frequency]:
        posit.append(select.positions)
        boxes.append(ts.dimensions[:3])
    
    #just convert to np array
    posit=np.array(posit)
    boxes=np.array(boxes)
    return posit,boxes

#using the 1 hot scheme to encode the sequence
#if the residue present, the index of list present 1 and other is 0

def OneHotEncoder(seqr):
    #turns sequence into matrix 1 hot encoding (list of vectors actually)
    #amino acid name
    aa_list=["ALA","ASN","CYS","ASP","GLU","PHE","GLY","HIE","ILE","LYS","LEU","MET","PRO","GLN","ARG","SER","THR","VAL","TRP","TYR"]#,"ACE","NME"]
    out=[]
    n_aa=len(aa_list)
    for aa in seqr:
        #set all 0
        tempa=np.zeros(n_aa)
        #replace some similar Residue
        if aa=="HIS" or aa=="HID":
            aa="HIE"
        if aa=="CYX" or aa=="CYM":
            aa="CYS"
        if aa=="ASH":
            aa="ASN"
        if aa=="GLH":
            aa="GLU"
        #set exist element index as 1da
        tempa[aa_list.index(aa)]=1.
        out.append(tempa)
    return out


#map_fg_cgt: 1D vector, length=number of atoms, value=index of residue for each atom belonged
#map_cg_fgt: 2D vector, [number of residue[...]] each vector includes the index of atom belong to one resiude
#res_encode: a matrix size residue number * residue type, one hot vector represent the type of residue
def make_cg_resolu(univ,cg_type):
    #right now cg_type= 1 residue per cg-bead
    if cg_type==1:
        #map_fg_cg[fg_i] = cg_i, cg bead index of fine grain index # 0 indexed
        #map_cg_fg[cg_i] = [fg_i,...], fg bead indexes of coarse grain index # 0 indexed
        #ingnore the modified residue ACE
        map_fg_cgt=[resi.resid-1 for resi in univ.select_atoms("all").residues for ai in resi.atoms if resi.resname !="ACE" ]
        map_cg_fgt=[[ai.index for ai in resi.atoms] for resi in univ.select_atoms("all").residues if resi.resname !="ACE" ]
        res_encode=OneHotEncoder([resi.resname for resi in univ.select_atoms("all").residues if resi.resname !="ACE" ])
    return map_fg_cgt,map_cg_fgt,res_encode


#cg_map: the same as map_cg_fgt
#aa_pos: 3D vector [number of frame[number of atom[x,y,z]]]
#cg_positis: 3D vector [number of frame[number of CG bead [x,y,z]]]
#aa_posits_out: 4D vector [number of frame[number of residues[number of atom per residue[x,y,z]]]]
def CGmap(map_cg_fgt,aa_pos):
    #assume molecules are not broken,
    #assumes Center of geometry
    cg_posits=[]
    aa_posits_out=[]
    for aa_posits in aa_pos:#loop over time
        #print(np.shape(aa_pos))
        cg_posits.append([np.average(aa_posits[aa_inds],axis=0) for aa_inds in map_cg_fgt])
        aa_posits_out.append([aa_posits[aa_inds] for aa_inds in map_cg_fgt])
    cg_posits=np.array(cg_posits)
    return cg_posits,aa_posits_out


#this function is simplely used to calculate the angle between three coordinates
def CalAngle(x1,x2,x3):
    #computes angle between x2-x1 and x3-x1
    v1=x2-x1
    v2=x3-x1
    v1l=math.sqrt(v1[0]*v1[0]+v1[1]*v1[1]+v1[2]*v1[2])
    v1n=np.divide(v1,v1l)
    v2l=math.sqrt(v2[0]*v2[0]+v2[1]*v2[1]+v2[2]*v2[2])
    v2n=np.divide(v2,v2l)
    #outputs in 2*radians/pi, range [0,1]
    return np.arccos(np.clip(np.dot(v1n, v2n), -1.0, 1.0))



#Calculate the distance between any 2 beads
#return 2D vector 
#3D cg postions
#system size, box
@jit(nopython=True)
def Dmap(sts):
    X,box=sts
    #box=box_size
    #compute pairwise distance matrix
    Nt= X.shape[0]
    M = X.shape[1]#number of beads
    N = X.shape[2]#xyz
    D = np.zeros((Nt,M, M))
    for iit in range(Nt):
        for iii in range(M):
            D[iit,iii, iii]=np.inf
            for jjj in range(iii+1,M):
                #loop over images of Y
                d_ims=np.zeros(27)#the distance to the 27 images
                ix=-1
                im_i=0
                for iix in range(3):
                    iy=-1
                    for iiy in range(3):
                        iz=-1
                        for iiz in range(3):
                            #compute distance between Atom[iii] and image of Atom[jjj]
                            d = 0.0
                            #computex,y,z
                            tmp = X[iit,iii, 0] - (X[iit,jjj, 0]+ix*box[iit,0])
                            d += tmp * tmp
                            tmp = X[iit,iii, 1] - (X[iit,jjj, 1]+iy*box[iit,1])
                            d += tmp * tmp
                            tmp = X[iit,iii, 2] - (X[iit,jjj, 2]+iz*box[iit,2])
                            d += tmp * tmp
                            d_ims[im_i]=np.sqrt(d)
                            im_i+=1
                            iz+=1
                        iy+=1
                    ix+=1
                #use the smallest distance (minimum image convention for PBC)
                mid_d=np.min(d_ims)
                D[iit,iii,jjj] = mid_d
                D[iit,jjj,iii] = mid_d
    return D


#check if 3 CG beads connected
def check_trip_bond(xxx):
    prmm,trjj,sel0,sel1,sel2=xxx
    univv=mda.Universe(prmm,trjj)
    #check if two residue bonded
    bonded_res01=univv.select_atoms("("+sel0+" and name C) and bonded "+"("+sel1+" and name N)")
    #check if two residue bonded
    bonded_res12=univv.select_atoms("("+sel1+" and name C) and bonded "+"("+sel2+" and name N)")
    if len(bonded_res01) > 0 and len(bonded_res12) > 0:
        return True
    else:
        return False
    


#num_CGbeads: number of CG beads
#n_proc: number of processes
#map_cg_fg: 2D CG map
#tri_seq_memb: 2D vector[number of trips[index of 3 CG beads]]
def build_trip(num_CGbeads,n_proc,prm,crd,map_cg_fg):
    stuff=[]
    possibles=[]
    cg_sels=["index "+" ".join([str(y) for y in x]) for x in map_cg_fg]
    for bead in range(num_CGbeads-2):#n_bead-2
        possible=[bead,bead+1,bead+2]
        stuff.append([prm,crd,cg_sels[possible[0]],cg_sels[possible[1]],cg_sels[possible[2]]])
        possibles.append(possible)
    with mp.Pool(n_proc) as P:
        tri_seq_membt=P.map(check_trip_bond,stuff)
    tri_seq_memb=[possibles[i] for i in range(num_CGbeads-2) if tri_seq_membt[i]]
    return tri_seq_memb


#pre calculte the data for a pdb file
#cg_pos: 2D vector, positions of CG bead for all frames
#the encode of CG sequence
#nn: number of nearest neighbors,pre-set 15
#pair_cg_dist: distance map between all beads for all frame
#cg_pos: 3D vector [number of frame[number of CG bead [x,y,z]]]
def preCalInput(pdb_id,unv,frenquency,tri_seq_memb,cg_pos,aa_pos,res_encode,pair_cg_dist,nn=15):

    #frame number 
    n_frame=len(unv.trajectory[::frenquency])
    
    # location of each bead in tripeptide tf.TensorSpec((3,3),dtype=tf.float32
    all_bead_loc=[]
    #tf.TensorSpec((3,3),dtype=tf.float32)
    all_bond_ang=[] 
    # tf.TensorSpec((3,3),dtype=tf.float32)
    all_trig_coord=[]
    #tf.TensorSpec((3,20),dtype=tf.float32)
    all_amino_encode=[]
    #tf.TensorSpec((3,15,20),dtype=tf.float32)
    all_close_amino_encode=[]
    #tf.TensorSpec((3,15),dtype=tf.float32)
    all_close_amino_dist=[]
    #tf.TensorSpec((3,26,25),dtype=tf.float32)
    all_invar_map=[]
    #tf.TensorSpec((3,26,25),dtype=tf.float32)
    all_tri_pep_cg=[]
    #tf.TensorSpec((3,27,3),dtype=tf.float32
    all_tri_pep_aa=[]





    #go through all frame
    for t in range(n_frame):

        # location of each bead in tripeptide tf.TensorSpec((3,3),dtype=tf.float32
        one_frame_all_bead_loc=[]
        #tf.TensorSpec((3,3),dtype=tf.float32)
        one_frame_all_bond_ang=[] 
        # tf.TensorSpec((3,3),dtype=tf.float32)
        one_frame_all_trig_coord=[]
        #tf.TensorSpec((3,20),dtype=tf.float32)
        one_frame_all_amino_encode=[]
        #tf.TensorSpec((3,15,20),dtype=tf.float32)
        one_frame_all_close_amino_encode=[]
        #tf.TensorSpec((3,15),dtype=tf.float32)
        one_frame_all_close_amino_dist=[]
        #tf.TensorSpec((3,26,25),dtype=tf.float32)
        one_frame_all_invar_map=[]
        #tf.TensorSpec((3,26,25),dtype=tf.float32)
        one_frame_all_tri_pep_cg=[]
        #tf.TensorSpec((3,27,3),dtype=tf.float32
        one_frame_all_tri_pep_aa=[]

        #go through the sequence
        for trip in tri_seq_memb:
            #indexs of three CG bead in one trip
            ind0,ind1,ind2=trip

            #determine the location of beads in triplet 
            #[[1,0,0] first bead
            # [0,1,0] second bead
            # [0,0,1]]  third bead
            bead_loc=np.eye(3)
            #add this to all_bead_loc
            one_frame_all_bead_loc.append(bead_loc)#print(bead_loc.shape)


            #determine two bonds(norma) and one angle of tri
            one_bond_ang=np.array([np.linalg.norm(cg_pos[t,ind1]-cg_pos[t,ind0]),np.linalg.norm(cg_pos[t,ind1]-cg_pos[t,ind2]),CalAngle(cg_pos[t,ind1],cg_pos[t,ind0],cg_pos[t,ind2])])
            #to keep consistence,three copies
            bond_ang=np.stack([one_bond_ang,one_bond_ang,one_bond_ang],axis=0)
            one_frame_all_bond_ang.append(bond_ang)
            

            #############this part does not record the center of particles
            #and the orientation of the triplet, so replace it with just positions

            #determine the coordinates of three beads
            #with the center bead locate at original point(0,0,0)
            #here relocate the center of 
            
            #the bond between ind0 and ind1 is on y-axis, z is 0 for all three points
            """
            trig_coord_a=np.array([0.,bond_ang[0,0],0.])
            trig_coord_b=np.array([0.,0.,0.])
            if bond_ang[0,2]>np.pi/2.:
                trig_coord_c=np.array([bond_ang[0,1]*np.cos(bond_ang[0,2]-(np.pi/2.)),-bond_ang[0,1]*np.sin(bond_ang[0,2]-(np.pi/2.)),0.])
            else:
                trig_coord_c=np.array([bond_ang[0,1]*np.cos((np.pi/2.)-bond_ang[0,2]),bond_ang[0,1]*np.sin((np.pi/2.)-bond_ang[0,2]),0.])
            
            #put all three coordinates into one vector
            trig_coord=np.stack([trig_coord_a,trig_coord_b,trig_coord_c])
            """
            #rewrite the triplet coordinates
            trig_coord_a=np.array(cg_pos[t][ind0])
            trig_coord_b=np.array(cg_pos[t][ind1])
            trig_coord_c=np.array(cg_pos[t][ind2])
            #print(trig_coord_a)
            #print(np.array([0.,0.,0.]))
            trig_coord=np.stack([trig_coord_a,trig_coord_b,trig_coord_c]) 

            one_frame_all_trig_coord.append(trig_coord)#print(trig_coord.shape)
            
 
            #encode the trip amino name with one hot vector
            amino_encode=np.stack([res_encode[ind0],res_encode[ind1],res_encode[ind2]],axis=0)
            #put everything into the list
            one_frame_all_amino_encode.append(amino_encode)#print(amino_encode.shape)


##########################
            #deal with the nearest neighbors

            close_amino_encode=[]
            
            #distance between the target CG bead and its nearest neighbor
            close_amino_dist=[]

            #calculate the invariance
        
            #
            invar_map=[]
            tri_pep_cg=[]
            tri_pep_aa=[]



            for ii in range(3):
                #the first nn number of nearest neighbors.
                #return a list of index of of top 15 nearest neighbors
                #np.argsort Returns the indices that would sort an array from small to large
                closest_N=np.argsort(pair_cg_dist[t,trip[ii]])[:nn]
                close_amino_encode.append([res_encode[x] for x in closest_N])

                #closest_N is the list of index, try to see if it works
                close_amino_dist.append(pair_cg_dist[t,trip[ii],closest_N])

                #calculate the invairance
                #the positon of CG bead ii in triplet
                tri_pep_cgt=cg_pos[t,trip[ii]]
                #the list of all atom position for each CG bead
                tri_pep_aat=aa_pos[t][trip[ii]]
                #the format of this two are not correct   
                #I am not sure how does it work,
                #but leave it here
                



                #max number of atoms in each residue
                #not sure why does it 26, but keep it here
                #the previous version is 26, still confusing
                nM=27
                #max number of atoms-number of atoms from pdb file
                n_diff_aa=nM-tri_pep_aat.shape[0]


                #fill others as 0
                zero=np.zeros((n_diff_aa,3))
                center=np.mean(tri_pep_aat,axis=0)

                print()
                for i in range(len(zero)):
                    zero[i]=center
                print(zero)
                tri_pep_aa.append(np.concatenate([tri_pep_aat,zero],axis=0))
                
                print("all")
                print(tri_pep_aa)
                #tri_pep_aa.append(np.concatenate([tri_pep_aat,np.zeros((n_diff_aa,3))],axis=0))
                

                #the number of atoms per residue
                N=tri_pep_aat.shape[0]
                #generate a single 2D vector 
                #if N=3-> [[-1,-1]]
                invar_mat=[-1*np.ones((1,N-1))]
                #append [[1,1],[1,1]]

                invar_mat.append(np.eye(N-1))
                #somewhat append commend still keek wired array formate
                #it's still a list of array, not a single matrix 
                #use concatenate to consistent the format
                invar_mat=np.concatenate(invar_mat,axis=0)
                
                #finally create a wired matrix
                #if N=3,nM=6
                #[[-1,-1,-1,0,0]
                # [1,0,0,0,0]
                # [0,1,0,0,0]
                # [0,0,1,0,0]
                # [0,0,0,0,0]
                # [0,0,0,0,0]]
                #except the first row, other are diogonal matrix 

                invar_mat=np.concatenate([invar_mat,np.zeros((n_diff_aa,N-1))],axis=0)
                invar_mat=np.concatenate([invar_mat,np.zeros((nM,nM-(N-1)-1))],axis=1)
                invar_map.append(invar_mat)
                
                


            

###########################
            
            #I didnot see the function of this command, but keep it here right now
            close_amino_encode=np.stack(close_amino_encode,axis=0)
            one_frame_all_close_amino_encode.append(close_amino_encode)
            
            #append the distance
            #print(np.shape(close_amino_dist))
            close_amino_dist=np.stack(close_amino_dist,axis=0)
            one_frame_all_close_amino_dist.append(close_amino_dist)#print(close_amino_dist.shape)

            #convert the array for 3 CG bead into a real matrix            
            invar_map=np.stack(invar_map,axis=0)
            one_frame_all_invar_map.append(invar_map)

            tri_pep_aa=np.stack(tri_pep_aa,axis=0)
            one_frame_all_tri_pep_aa.append(tri_pep_aa)



        all_bead_loc.append(one_frame_all_bead_loc)

        all_bond_ang.append(one_frame_all_bond_ang) 
        all_trig_coord.append(one_frame_all_trig_coord)
        all_amino_encode.append(one_frame_all_amino_encode)
    #tf.TensorSpec((3,15,20),dtype=tf.float32)
        all_close_amino_encode.append(one_frame_all_close_amino_encode)
    #tf.TensorSpec((3,15),dtype=tf.float32)
        all_close_amino_dist.append(one_frame_all_close_amino_dist)

        all_invar_map.append(one_frame_all_invar_map)

        all_tri_pep_aa.append(one_frame_all_tri_pep_aa)



        #save the data into files

    out_root="/home/yu/Desktop/mutiscale/outdata/"+pdb_id+"/"+pdb_id
    out_dir="/home/yu/Desktop/mutiscale/outdata/"+pdb_id+"/"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    #print(len(all_amino_encode))
    np.save(out_root+"_all_bead_loc_"+str(frenquency)+".npy",all_bead_loc)
    np.save(out_root+"_all_bond_ang_"+str(frenquency)+".npy",all_bond_ang)
    np.save(out_root+"_all_trig_coord_"+str(frenquency)+".npy",all_trig_coord)
    np.save(out_root+"_all_amino_encode_"+str(frenquency)+".npy",all_amino_encode)
    np.save(out_root+"_all_close_amino_encode_"+str(frenquency)+".npy",all_close_amino_encode)
    np.save(out_root+"_all_close_amino_dist_"+str(frenquency)+".npy",all_close_amino_dist)
    np.save(out_root+"_all_invar_map_"+str(frenquency)+".npy",all_invar_map)
    #np.save(out_root+"_all_tri_pep_cg_"+str(frenquency)+".npy",all_tri_pep_cg)
    np.save(out_root+"_all_tri_pep_aa_"+str(frenquency)+".npy",all_tri_pep_aa)


        #check if the data saved correctly
    Lall_bead_loc=np.load(out_root+"_all_bead_loc_"+str(frenquency)+".npy")
    print(np.shape(Lall_bead_loc)) 
    if False==np.array_equal(all_bead_loc,Lall_bead_loc):
        print("some error in all_bead_loc") 

    Lall_bond_ang=np.load(out_root+"_all_bond_ang_"+str(frenquency)+".npy")
    print(np.shape(Lall_bond_ang)) 
    if False==np.array_equal(Lall_bond_ang,all_bond_ang):
        print("some error in all_bond_ang") 

    Lall_tri_coord=np.load(out_root+"_all_trig_coord_"+str(frenquency)+".npy")
    print(np.shape(Lall_tri_coord)) 
    if False==np.array_equal(all_trig_coord,Lall_tri_coord):
        print("some error in all_tri_coord") 


    Lall_amino_encode=np.load(out_root+"_all_amino_encode_"+str(frenquency)+".npy") 
    print(np.shape(Lall_amino_encode))
    if False==np.array_equal(all_amino_encode,Lall_amino_encode):
        print("some error in all_amino_encode") 


    Lall_close_amino_dist=np.load(out_root+"_all_close_amino_dist_"+str(frenquency)+".npy")
    print(np.shape(Lall_close_amino_dist)) 
    if False==np.array_equal(all_close_amino_dist,Lall_close_amino_dist):
        print("some error in all_close_amino_dist")


    Lall_close_amino_encoder=np.load(out_root+"_all_close_amino_encode_"+str(frenquency)+".npy")
    print(np.shape(Lall_close_amino_encoder))
    if False==np.array_equal(Lall_close_amino_encoder,all_close_amino_encode):
        print("some error in all_close_amino_encode")


    Lall_invar_map=np.load(out_root+"_all_invar_map_"+str(frenquency)+".npy")
    print(np.shape(Lall_invar_map))
    if False==np.array_equal(Lall_invar_map,all_invar_map):
        print("some error in all_invar_map")


    Lall_trip_pep_all=np.load(out_root+"_all_tri_pep_aa_"+str(frenquency)+".npy")
    print(np.shape(Lall_trip_pep_all))
    if False==np.array_equal(Lall_trip_pep_all,all_tri_pep_aa):
        print("some error in all_trip_pep_aa")

    print(pdb_id+"is successfully saved.")



def get_all_pdbs(path="/home/yu/Desktop/mutiscale/outdata/"):

    """
    all_pdbs=["12E8","1A3H","1A4U","1A7U","1A88","1AGJ","1AKO","1ARL","1AUO","1BEC","1BHE","1BKP","1BQC","1BUE",  
        "1BYI","1CNS","1CZ1","1DIX","1DJA","1DQ0","1DUZ","1EDG","1EQP","1ERZ","1F00","1F5Z","1FBA","1FSF",  
        "1FTR","1G24","1G8A","1GQN","1GVL","1GZJ","1H6U","1HRD","1HTR","1HXH","1HYL","1I7U","1IDK","1IJB",  
        "1IU8","1JFL","1JGV","1JLN","1KCV","1KS9","1L7A","1L8F","1LBV","1LP8","1LU9","1M5S","1M6O","1MIZ",  
        "1MMI","1N7P","1NBZ","1NGZ","1NLB","1OA4","1OCK","1OJQ","1OMP","1ONR","1ORS","1P3C","1PE9","1PIP",  
        "1PP3","1PXZ","1PZ5","1QCX","1QTS","1RC9","1RL0","1T06","1T4D","1THV","1TIB","1TJE","1TUX","1U00",  
        "1UGH","1USG","1VAX","1X1E","1XDW","1XH3","1XSZ","1XVM","1Y6I","1YPI","1YT4","1YUO","1Z15","1ZAH",  
        "1ZHL","1ZSD","1ZZG","2A6Z","2AJU","2B5R","2BAA","2BKR","2BNU","2CB5","2CGA","2CLZ","2CV3","2CYG",  
        "2D0I","2D5J","2DUC","2ERF","2EXO","2EYI","2FAT","2FB4","2FZ3","2G2U","2G5X","2GAS","2GGO","2H2Z",  
        "2H6P","2HAD","2HJK","2HLC","2HOB","2HVM","2IE8","2LAO","2NW3","2O6S","2OF3","2OKT","2OMZ","2OR2",  
        "2PA6","2PET","2QCY","2QHT","2SIL","2V2W","2VLL","2VU8","2WBZ","2WPL","2X8X","2XFX","2YMU","2YNO",  
        "2YPK","2YXP","3AAP","3APP","3BWA","3CTK","3D25","3ER5","3F7M","3GSW","3H6J","3ILS","3KLA","3KPQ",  
        "3LIG","3LKO","3LN5","3LY3","3M4D","3M66","3N4I","3OT7","3OXR","3P8T","3PTL","3PWU","3RP2","3RRS",  
        "3SEB","3TUA","3UTQ","3VFS","3W3E","3W4Q","3WA1","3WP5","3WY8","3X13","4AXU","4CFI","4DIY","4DJ5",  
        "4FCU","4G5Z","4G6K","4GKU","4GUZ","4H20","4I4N","4J4R","4JCN","4JJO","4JZC","4KEL","4L9R","4LE8",  
        "4LSW","4MCK","4MHP","4NT6","4O2C","4OZX","4P9N","4PBO","4PMH","4PR5","4R1N","4RVS","4RXV","4TX7",  
        "4V38","4WJS","4WUM","4X7S","4XIO","4YHE","4ZPB","4ZTP","5A8U","5AMZ","5AVG","5C0D","5DDT","5DJ7",  
        "5DK1","5DTX","5DZ9","5E5B","5EFS","5EO1","5EWT","5GLX","5GN2","5GQP","5GS7","5GY3","5H0Q","5H28",  
        "5H4E","5H5Z","5HHP","5IB1","5IHW","5J2V","5KEH","5KJV","5MAL","5ORI","5OYX","5OZ9","5P00","5P11",  
        "5P20","5P31","5P42","5P53","5P64","5P75","5P86","5TUN","5U3P","5UCB","5VTL","5VWH","5XCY","5XOS",  
        "5Y33","5YMW","5YNS","5YPV","5YR3","5YSX","5YUP","5Z2D","5ZFH","6A41","6AFM","6CL7","6D0A","6D29",  
        "6E3D","6E4D","6F4M","6FAB","6GFV","6GGP","6HY2","6I3B","6IAS","6IM4","6J1W","6JTP","6KGJ","6L8T",  
        "6NZS","6OKJ","6PKP","6PYW","6PZ5","6QE3","6VT3","6WTM","6XIA","6Y2E","7AHL"]
    """

    #issue happens at 3ER5, beyond that, the is
    all_pdbs=["12E8","1A3H","1A4U","1A7U","1A88","1AGJ","1AKO","1ARL","1AUO","1BEC","1BHE","1BKP","1BQC","1BUE"]
        #      "1BYI","1CNS","1CZ1","1DIX","1DJA","1DQ0","1DUZ","1EDG","1EQP","1ERZ","1F00","1F5Z","1FBA","1FSF",  
        #"1FTR","1G24","1G8A","1GQN","1GVL","1GZJ","1H6U","1HRD","1HTR","1HXH","1HYL","1I7U","1IDK","1IJB",  
        #"1IU8","1JFL","1JGV","1JLN","1KCV","1KS9","1L7A","1L8F","1LBV","1LP8","1LU9","1M5S","1M6O","1MIZ",]
        #"1MMI","1N7P","1NBZ","1NGZ","1NLB","1OA4","1OCK","1OJQ","1OMP","1ONR","1ORS","1P3C","1PE9","1PIP",  
        #"1PP3","1PXZ","1PZ5","1QCX","1QTS","1RC9","1RL0","1T06","1T4D","1THV","1TIB","1TJE","1TUX","1U00",  
        #"1UGH","1USG","1VAX","1X1E","1XDW","1XH3","1XSZ","1XVM","1Y6I","1YPI","1YT4","1YUO","1Z15","1ZAH",  
        #"1ZHL","1ZSD","1ZZG","2A6Z","2AJU","2B5R","2BAA","2BKR","2BNU","2CB5","2CGA","2CLZ","2CV3","2CYG"]
      
        #,"2D0I","2D5J","2DUC","2ERF","2EXO","2EYI","2FAT","2FB4","2FZ3","2G2U","2G5X","2GAS","2GGO","2H2Z",
        #"2H6P","2HAD","2HJK","2HLC","2HOB","2HVM","2IE8","2LAO","2NW3","2O6S","2OF3","2OKT","2OMZ","2OR2",  
        #"2PA6","2PET","2QCY","2QHT","2SIL","2V2W","2VLL","2VU8","2WBZ","2WPL","2X8X","2XFX","2YMU","2YNO"]
    
    #all_pdbs=["12E8"]
    
    all_pdbst=[]
    for pdb in all_pdbs:
        if os.path.isdir(path+pdb+"/"):
        #if os.path.exists("/gpfs3/scratch/jremingt/tf_trainingset_4_7_2022/"+pdb+"_all_tri_pep_cg_"+str(tmodload)+".npy"):
            all_pdbst.append(pdb)
    return all_pdbst

def loadnpy(path_file):
    return np.load(path_file)

def preloadData():
    pass


#n_B:number of beads per segment
#H:number of symetry function per bead type
#L=4

#n_N: the number of nearest neighbor per atom
#n_M: the max number of atoms for each CG bead
#n_B: the number of CG bead per segment
def InitialInput(n_B=3,n_N=15,n_M=26):
    inp_bead_loc=tf.keras.Input(shape=(n_B,3))
    inp_bond_ang=tf.keras.Input(shape=(n_B,3))
    inp_trig_coord=tf.keras.Input(shape=(n_B,3))
    inp_amino_encode=tf.keras.Input(shape=(n_B,20))
    inp_close_amino_encode=tf.keras.Input(shape=(n_B,n_N,20))
    inp_close_amino_dist=tf.keras.Input(shape=(n_B,n_N))
    inp_invar_map=tf.keras.Input(shape=(n_B,n_M,n_M-1))

    return inp_bead_loc,inp_bond_ang,inp_trig_coord,inp_amino_encode,inp_close_amino_encode,inp_close_amino_dist,inp_invar_map




#n_en_lay: number of encode layer
def EncodeBeadType(n_en_lay=3,fpt_ws=[],nl_act_G=tf.keras.layers.LeakyReLU(alpha=0.2),inp_amino_encode=tf.keras.Input(shape=(3,20)),inp_close_amino_encode=tf.keras.Input(shape=(3,15,20)),dropout=.15):
    for i in range(n_en_lay):
        if i ==0:
            #Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as #the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if #use_bias is True). These are all attributes of Dense.
            #A regularizer that applies a L2 regularization penalty.
            #The L2 regularization penalty is computed as: loss = l2 * reduce_sum(square(x))
            
            layered=tf.keras.layers.Dense(fpt_ws[i],activation=nl_act_G,use_bias=True,kernel_regularizer= tf.keras.regularizers.L2(1e-2))
        else:
            layered=tf.keras.layers.Dense(fpt_ws[i],activation=nl_act_G,use_bias=True)
        if i==0:
            #pass bead_i through it
            bead_i_out=layered(inp_amino_encode)
            #pass neighbors through it
            bead_n_out=layered(inp_close_amino_encode)
        
        else:
            #randomly dropped the time 
            #The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. 
            #Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged.

            bead_i_out=tf.keras.layers.Dropout(dropout)(bead_i_out)
            bead_i_out=tf.keras.layers.Dropout(dropout)(bead_i_out)
            bead_i_out=layered(bead_i_out)
            bead_n_out=layered(bead_n_out)
    
    #encode of triplet aminos
    bead_i_out=tf.keras.layers.Softmax()(bead_i_out)

    #encode of neighboring aminos
    bead_n_out=tf.keras.layers.Softmax()(bead_n_out)

    return bead_i_out,bead_n_out


def radialSysmetryFunction(inp_close_amino_dist,all_close_amino_dist,bead_i_out,bead_n_out,H=24,L=4,r_cut=15,n_B=3,plot=False):
    #the center of gaussian
    r_Hs=np.linspace(3.5,r_cut+.5,H)#12
    #print("r_Hs+ "+ str(r_Hs))
    #number of histogram
    #the wideth of gaussian
    n_Hs=(2*(r_cut+0.5-3.5)/H)*np.ones((H))#2
    #print("n_Hs+ "+str(n_Hs))
        #### radial symetry functions
    
    if plot:
        #plot symmetry functions
        #0.15,0.3,0.45...
        r_space=np.linspace(0,15,100)
        for h in range(H):
            plt.plot(r_space,tf.where(r_space<r_cut,np.exp(-n_Hs[h]*np.square(r_space-r_Hs[h]))*0.5*(np.cos(np.pi*r_space/r_cut)+1.),0)/2)
        
        #a lot 0 because of distance between itself
        plt.hist(all_close_amino_dist[all_close_amino_dist<r_cut].flatten(),bins=200,density=True)
    
        plt.savefig("symmetry functions")
    
    #compute gaussian functions (N,H)
    gaus_funs=[]
    for h in range(H):
        #gaus_funs.append(tf.math.exp(-n_Hs[h]*tf.math.square(input_dists-r_Hs[h]))*0.5*tf.where(input_dists<r_cut,tf.math.cos(np.pi*input_dists/r_cut)+tf.ones([input_dists.shape[1],input_dists.shape[2]]),tf.zeros([input_dists.shape[1],input_dists.shape[2]])))
        gaus_funs.append(tf.math.exp(-n_Hs[h]*tf.math.square(inp_close_amino_dist-r_Hs[h])))
        #gaus_funs.append(np.exp(-n_Hs[h]*np.square(input_dists-r_Hs[h]))*0.5*(np.cos(np.pi*input_dists/r_cut)+1.))
    gaus_funs=tf.stack(gaus_funs,axis=2)
    #print("gaus_funs",gaus_funs.shape)
    #exit()
    all_gaus_funs=[]
    for l in range(L):
        for k in range(l,L):
            all_gaus_funs.append(gaus_funs)
    all_gaus_funs=tf.stack(all_gaus_funs,axis=2)
    
    pair_fpts=[]
    #loop over pair fpts and compute weights and expanded sym funcs
    for l in range(L):
        for k in range(l,L):
            #print("bead_i_out",bead_i_out.shape)
            #print("bead_n_out",bead_n_out.shape)
            fpt_i=tf.expand_dims(tf.expand_dims(bead_i_out[:,:,l],axis=2),axis=-1)
            sym_weight=tf.expand_dims(bead_n_out[:,:,:,k],axis=2)#tf.expand_dims(tf.expand_dims(,axis=2),axis=-1)
            sym_funcs=tf.multiply(sym_weight,gaus_funs)
            #Partially-known shape: has a known number of dimensions, and an unknown size for one or more dimension. e.g. TensorShape([None, 256])
            #print(sym_weight)
            #print("sym_weight",sym_weight.shape)
            #print("sym_funcs",sym_funcs.shape)
            pair_fpts.append(sym_funcs)
            #exit()
        #pair_fpts.append(tf.expand_dims(bead_i_out[:,:,l],axis=2))
    
    pair_fpts=tf.stack(pair_fpts,axis=2)
    #print("pair_fpts",pair_fpts.shape)
    #exit()
    #apply sum over neighbors
    sym_funs=tf.reduce_sum(pair_fpts,axis=4)
    
    flat_sym_funs=tf.reshape(sym_funs,(tf.shape(sym_funs)[0],n_B,sym_funs.shape[2]*sym_funs.shape[3]))
    #print(np.shape(flat_sym_funs))
    return flat_sym_funs

@tf.function
def align_loss(y_true, y_pred):
    #compute R that rotates the COG of y_pred with COG of y_true (average along 1st and 2nd axis)
    #index=tf.reduce_sum(y_true,axis=3)!=0.#).shape)
    n_M=26
    n_B=3
    
    print(n_M)
    print(y_true,y_pred)

    #get CG of y_true, and y_pred
    y_true_cg=y_true[:,:,n_M]
    y_pred_cg=y_pred[:,:,n_M]
    y_pred_aa=y_pred[:,:,:n_M]
    y_true_aa=y_true[:,:,:n_M]
    print("get mean")
    cg_trans_true=tf.reduce_mean(y_true_cg,axis=1,keepdims=True)
    cg_trans_pred=tf.reduce_mean(y_pred_cg,axis=1,keepdims=True)

    print("Computes the sum of elements across dimensions of a tensor.")  
    index=tf.reduce_sum(tf.cast(y_true_aa!=0.,tf.float32),axis=2)

    ##~https://en.wikipedia.org/wiki/Kabsch_algorithm
    #a method for calculating the optimal rotation matrix that minimizes the RMSD (root mean squared deviation) between two paired sets of points.
    #The algorithm works in three steps: a translation, the computation of a covariance matrix, and the computation of the optimal rotation matrix.
    print("recenter everthing")

    #Translation
    #Both sets of coordinates must be translated first, so that their centroid coincides with the origin of the coordinate system. 
    #This is done by subtracting from the point coordinates of the respective centroid.
    true_mat=y_true_cg-cg_trans_true
    pred_mat=y_pred_cg-cg_trans_pred

    #Computation of the covariance matrix
    #The second step consists of calculating a matrix H. In matrix notation,
    print("Computation of the covariance matrix")
    H=tf.linalg.matmul(pred_mat, true_mat, transpose_a=True, transpose_b=False)

    ##H a is a tensor.
    # s is a tensor of singular values.
    # u is a tensor of left singular vectors.
    # v is a tensor of right singular vectors.
    print("1")
    s, u, v = tf.linalg.svd(H)
    D=tf.math.sign(tf.linalg.det(tf.linalg.matmul(v, u,transpose_b=True)))#))transpose_a=False, transpose_b=True)))
    print("2")
    Dt=tf.map_fn(fn=lambda t: tf.concat([tf.ones(2,dtype=tf.float32),[t]],axis=0), elems=D)
    print("2.5")
    print(y_pred.shape[-1])
    #print(tf.shape(y_pred)[0])
    #a=tf.eye(y_pred.shape[-1],dtype=tf.float32)
    #print(a)
    #np_y_pred = y_pred.numpy()
    Dtt=tf.eye(y_pred.shape[-1],batch_shape=[tf.shape(y_pred)[0]],dtype=tf.float32)
    #Dtt=tf.eye(y_pred.shape[-1],batch_shape=[tf.shape(y_pred)[0]],dtype=tf.float32)
    print("31")
    Dttt=tf.cast(tf.linalg.set_diag(Dtt,Dt),tf.float32)
    #calculate rotation matrix
    R=tf.linalg.matmul(tf.linalg.matmul(v,Dttt),u, transpose_b=True)
    #copy R
    Rs=[]
    for idx in range(n_B):
        Rs.append(R)
    Rs=tf.stack(Rs,axis=1)

    ###test R
    #print(pred_mat.shape)
    #print(R.shape)
    #align_cg=tf.matmul(pred_mat,R,transpose_b=True)
    #print(true_mat[0])
    #print(align_cg[0])
    #exit()
    print("4")
    cg_trans_true=tf.expand_dims(cg_trans_true,axis=1)
    cg_trans_pred=tf.expand_dims(cg_trans_pred,axis=1)
    #shift aa_pred
    shift_aa_pred=y_pred_aa-cg_trans_pred
    #apply alignment
    align_aa=tf.matmul(shift_aa_pred,Rs,transpose_b=True)
    #translate by true CG
    align_trans=tf.add(align_aa,cg_trans_true)
    align_trans=tf.where(y_true_aa!=0.,align_trans,tf.zeros_like(y_true_aa))
    squared_disp=tf.reduce_sum(tf.square(tf.subtract(y_true_aa, align_trans)),axis=-1)
    mean_squared_disp=tf.reduce_sum((1./index[:,:,0:1])*squared_disp,axis=2)#MSD for each residue
    per_res_rmsd=tf.sqrt(mean_squared_disp)
    return tf.reduce_mean(per_res_rmsd)

def align_pred(y_true, y_pred):
    #compute R that rotates the COG of y_pred with COG of y_true (average along 1st and 2nd axis)
    #index=tf.reduce_sum(y_true,axis=3)!=0.#).shape)
    n_M=26
    n_B=3
    #get CG of y_true, and y_pred
    y_true_cg=y_true[:,:,n_M]
    y_pred_cg=y_pred[:,:,n_M]
    y_pred_aa=y_pred[:,:,:n_M]
    y_true_aa=y_true[:,:,:n_M]
    cg_trans_true=tf.reduce_mean(y_true_cg,axis=1,keepdims=True)
    cg_trans_pred=tf.reduce_mean(y_pred_cg,axis=1,keepdims=True)

    index=tf.reduce_sum(tf.cast(y_true_aa!=0.,tf.float32),axis=2)

    ##~https://en.wikipedia.org/wiki/Kabsch_algorithm
    true_mat=y_true_cg-cg_trans_true
    pred_mat=y_pred_cg-cg_trans_pred

    H=tf.linalg.matmul(pred_mat, true_mat, transpose_a=True, transpose_b=False)
    s, u, v = tf.linalg.svd(H)
    D=tf.math.sign(tf.linalg.det(tf.linalg.matmul(v, u,transpose_b=True)))#))transpose_a=False, transpose_b=True)))
    Dt=tf.map_fn(fn=lambda t: tf.concat([tf.ones(2,dtype=tf.float32),[t]],axis=0), elems=D)
    Dtt=tf.eye(y_pred.shape[-1],batch_shape=[tf.shape(y_pred)[0]],dtype=tf.float32)
    Dttt=tf.cast(tf.linalg.set_diag(Dtt,Dt),tf.float32)
    #calculate rotation matrix
    R=tf.linalg.matmul(tf.linalg.matmul(v,Dttt),u, transpose_b=True)
    #copy R
    Rs=[]
    for idx in range(n_B):
        Rs.append(R)
    Rs=tf.stack(Rs,axis=1)

    ###test R
    #print(pred_mat.shape)
    #print(R.shape)
    #align_cg=tf.matmul(pred_mat,R,transpose_b=True)
    #print(true_mat[0])
    #print(align_cg[0])
    #exit()
    cg_trans_true=tf.expand_dims(cg_trans_true,axis=1)
    cg_trans_pred=tf.expand_dims(cg_trans_pred,axis=1)
    #shift aa_pred
    shift_aa_pred=y_pred_aa-cg_trans_pred
    #apply alignment
    align_aa=tf.matmul(shift_aa_pred,Rs,transpose_b=True)
    #translate by true CG
    align_trans=tf.add(align_aa,cg_trans_true)
    return align_trans


def savetoXYZ(coordlist,fileName):
    try:
        os.remove(fileName)
    except OSError:
        pass
    

    pos1=coordlist[0][0]
    print(pos1)
    pos2=coordlist[0][1]
    pos3=coordlist[0][2]

    f=open(fileName,"a")
    f.write(str(len(pos1)+len(pos2)+len(pos3))+"\n")
    f.write("txt\n")
    
    pos1=coordlist[0][0]
    print(pos1)
    pos2=coordlist[0][1]
    pos3=coordlist[0][2]

    #for i in range(len(coordlist[0])):
    #    f.write(str("1")+"  "+str(coordlist[0][i][0])+"   "+str(coordlist[0][i][1])+"   "+str(coordlist[0][i][2])+"\n")

    for i in range(len(pos1)):
        f.write(str("1")+"  "+str(pos1[i][0])+"   "+str(pos1[i][1])+"   "+str(pos1[i][2])+"\n")
    
    for i in range(len(pos2)):
        f.write(str("2")+"  "+str(pos2[i][0])+"   "+str(pos2[i][1])+"   "+str(pos2[i][2])+"\n")

    for i in range(len(pos3)):
        f.write(str("3")+"  "+str(pos3[i][0])+"   "+str(pos3[i][1])+"   "+str(pos3[i][2])+"\n")

    f.close



def saveCGtoXYZ(coordlist,fileName):
    try:
        os.remove(fileName)
    except OSError:
        pass
    
    
    print(coordlist)


    f=open(fileName,"a")
    f.write(str(len(coordlist[0]))+"\n")
    f.write("txt\n")


    for i in range(len(coordlist[0])):
        f.write(str("1")+"  "+str(coordlist[0][i][0])+"   "+str(coordlist[0][i][1])+"   "+str(coordlist[0][i][2])+"\n")

    f.close
