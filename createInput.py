import functions as func
import readNC 
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt


def generatebeadloc():
    all_bead_loc=[]
    


# a simple function to perform test
def test():
    crd=data_dir+"/12E8/12E8_stripped_with_force.nc"
    prm=data_dir+"/12E8/12E8_stripped.top"
    uni=readNC.readNC(prm,crd)

    #simple test 

    residue_target="ALA"
    
    target_res_sel=[x.atoms for x in uni.select_atoms('resname '+residue_target).residues]
    target_res_i=[x.resid-1 for x in uni.select_atoms('resname '+residue_target).residues]#starts on 1 so -1

    #print(target_res_i)

    
    N=10
    nM=15
    n_diff_aa=15-10

    invar_mat=[-1*np.ones((1,N-1))]
    print(invar_mat)
                #append [[1,1],[1,1]]
    invar_mat.append(np.eye(N-1))
    print(invar_mat)
    invar_mat=np.concatenate(invar_mat,axis=0)
    print(invar_mat)

    invar_mat=np.concatenate([invar_mat,np.zeros((n_diff_aa,N-1))],axis=0)
    print(invar_mat)
    invar_mat=np.concatenate([invar_mat,np.zeros((nM,nM-(N-1)-1))],axis=1)
    print(invar_mat)
    print(np.shape(invar_mat))
    #invar_map.append(invar_mat)


    #print(uni)
    f=10
    pos,box=func.get_posits(univ=uni,frequency=f)
    
    #print(len(pos))
    #print(len(box))
    
    a,b,c=func.make_cg_resolu(univ=uni,cg_type=1)
    #print(len(a))
    print(len(b))
    #print(b)
    #print(a)
    
    print("Finding Tripeptides on same protein chains")
    
    tri=func.build_trip(len(b),20,prm,crd,b)
    print(len(tri))
    print(tri)


    bond_ang=np.array([0,1,0])
    bond_ang=np.stack([bond_ang,bond_ang,bond_ang],axis=0)

    cgp,aap=func.CGmap(b,pos)
    
    print(np.shape(cgp))
    #print(len(aap))
    #print(len(aap[0]))
    #print(len(aap[0][0]))
    #print(len(aap[0][0][0]))
    #print(aap)
    
    print(np.eye(3))

    #calculate the pair distance[number of frame[number of CG bead, number of CG bead]]
    print("calculate the pair distance between twos")


    #number of CPU cores can parallas
    num_thread=22
    stuff=zip(np.array_split(cgp,num_thread),np.array_split(box,num_thread))
    with mp.Pool(num_thread) as P:
        pair_cg_dist=P.map(func.Dmap,stuff)
    pair_cg_dist=np.concatenate(pair_cg_dist,axis=0)
    pair_cg_dist[pair_cg_dist == np.inf] = 0
    
    
    
    #print(np.shape(pair_cg_dist))
    #print(pair_cg_dist[0])
    #plt.hist(pair_cg_dist.flatten())
    #plt.savefig("test.png")
    #plt.show()

    #D=func.Dmap(pos=cgp,box_size=box)
    #print(D)
    #print(np.shape(D))


if __name__=="__main__":

    
    #data_dir="/home/yu/10TB/jacob/CG_files/trajectories"

    data_dir="/home/yu/10TB/CG_SIM/cg_stripped_trj"

    out_dir="/home/yu/Desktop/mutiscale/outdata"
    
    #1F5Z: no trj file

    all_pdbs=["12E8","1A3H","1A4U","1A7U","1A88","1AGJ","1AKO","1ARL","1AUO","1BEC","1BHE","1BKP","1BQC","1BUE",  
        "1BYI","1CNS","1CZ1","1DIX","1DJA","1DQ0","1DUZ","1EDG","1EQP","1ERZ","1F00","1FBA","1FSF","1F5Z",  
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
        
    print("total number of pdb files: "+ str(len(all_pdbs)))
    
    testpdb=["12E8"]
    
    #for pdb in testpdb:
    for pdb in all_pdbs:
        
        
        #read the data of pdb into univ 
        crd=data_dir+"/"+pdb+"/"+pdb+"_stripped_with_force.nc"
        prm=data_dir+"/"+pdb+"/"+pdb+"_stripped.top"
        uni=readNC.readNC(prm,crd)
        print(pdb+" read.")

        #the frequency of loaded frame
        frequency=1000
        print(str(1000)+" frequency set .")



        #the positon of all atom, and its corresponded box size
        pos,box=func.get_posits(univ=uni,frequency=frequency)

        print("postion and box size read.")

        
        #return the map between all atom and cg beds and residue encode using one hot encoder scheme
        map_fg_cgt,map_cg_fgt,res_encode=func.make_cg_resolu(univ=uni,cg_type=1)

        print("CG map generated")

        #return the CG bead position and allatom position
        cgp,aap=func.CGmap(map_cg_fgt,pos)

        print("CG map applied.")

        #return the encoder triplet
        tri=func.build_trip(len(map_cg_fgt),22,prm,crd,map_cg_fg=map_cg_fgt)
            #number of CPU cores can parallas
        
        print("triplet calculated")

        num_thread=22
        stuff=zip(np.array_split(cgp,num_thread),np.array_split(box,num_thread))
        with mp.Pool(num_thread) as P:
            pair_cg_dist=P.map(func.Dmap,stuff)
        pair_cg_dist=np.concatenate(pair_cg_dist,axis=0)
        pair_cg_dist[pair_cg_dist == np.inf] = 0

        print("pair maps calculated"+str(np.shape(pair_cg_dist)))

        func.preCalInput(pdb_id=pdb,unv=uni,frenquency=frequency,tri_seq_memb=tri,cg_pos=cgp,aa_pos=aap,res_encode=res_encode,pair_cg_dist=pair_cg_dist, nn=15)


    #test()
    
    
    