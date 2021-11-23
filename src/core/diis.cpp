#include <occ/core/diis.h>

namespace occ::core::diis {

Mat DIIS2::commutator(const Mat &A, const Mat &B, const Mat &overlap) {
    return  A * B * overlap - overlap * B * A;
}

/*

Mat DIIS2::extrapolate_roothaan(const Mat &F, const Mat &comm) {

}

Mat DIIS2::extrapolate_restart(const Mat &F, const Mat &comm) {
}

Mat DIIS2::extrapolate_adaptive_depth(const Mat &F, const Mat &comm) {
}

Mat DIIS2::extrapolate_fixed_depth(const Mat &F, const Mat &comm) {
}

Mat DIIS2::extrapolate(const Mat &F, const Mat &comm) {

}
*/
/*
    def (dm0,mf,mol,tol=1e-08,maxiter=50,mode="R-CDIIS",
         adaptative=False, sizediis=8,cstAdapt=1e-05, minrestart = 1,
         modeQR="full",param=0.1,booldiis=True,slidehole=False,log=logging.DEBUG,name=""):
*/
/*
        CDiis algorithm
        --------------
                
        CDIIS algorithm. Multiple variations of the algorithm are implemented. 
        The core of the function is to implement the restarted CDIIS (R-CDIIS)
        and the Adaptive-Depth CDIIS (AD-CDIIS) compared to the Fixed-Depth CDIIS (FD-CDIIS)

        Parameters 
        ----------
        dm0 : numpy.array
            initialization of the density matrix
        mf : object of the chosen pyscf class (for instance scf.hf.RHF)
        mol : object of the molecule pyscf class (gto.Mole())
        param : float
            default value : 0:1
            tau parameter for the R-CDIIS algorithm
            delta parameter for the AD-CDIIS algorithm
        tol : float
            default value : 1e-08
            tolerence parameter for convergence test on residual (commutator)
        maxiter : integer
            default value : 50
        maximal number of iterations allowed
        mode : string 
        default value : "R-CDIIS"
            four modes available : "R-CDIIS", "AD-CDIIS", "FD-CDIIS", "Roothaan"
        adaptative : boolean
            default value : False
            adaptative mode for the tau parameter
        sizediis : integer
            default value : 8
            size of the window of stored previous iterates in the FD-CDIIS algorithm 
            this dimension is also used for the adaptative algorithm 
        cstAdapt : float
            default value : 1e-05
            constant part for the adaptative parameter (delta or tau)
        minrestart : integer
            default value : 1
            number of iterates we keep when a restart occurs
        modeQR : string 
            default value : "full"
            mode to build the QR decomposition of the matrix of residuals differences
            - full to compute the qr decomposition with scipy.linalg.qr
            - economic to use the economic mode of scipy.linalg.qr 
            - otherwise : compute the qr decomposition with scipy.linalg.qr_insert and scipy.linalg.qr_delete
        slidehole : boolean
            default value : False
            if True : allows hole in the AD-CDIIS algorithm
        name : string
            default value "" 
            name of the computation (to identify the log file : diis_namevalue.log)
        
        Outputs
            -------
        energy : float 
            final energy after convergence
        conv : boolean
            if convergence, True, else, False 
        rnormlist : numpy.array
            list of norm of r_k
        mklist : numpy.array
            list of m_k value at each step
        cnormlist : numpy.array
            list of the iterates of the norm of the c_i coefficients 
        dmlast : numpy.array
            last computed density matrix  
        booldiis : boolean
            default value : True
            if False : Roothann algorithm
            if True : one of the CDIIS algorithm
        */
 /*       
        if(mode=="Roothaan"):
            booldiis == False;
        else:
            booldiis == True;
                                    
        // compute the initial values
        h1e = mf.get_hcore(mol) # core hamiltonian
        s1e = mf.get_ovlp(mol) # overlapping matrix
        vhf0 = mf.get_veff(mol, dm0) # potential
        fock0 = mf.get_fock(h1e, s1e, vhf0, dm0, 0, None, 0, 0, 0) 

        // Aufbau
        dm = Aufbau(fock0, s1e, mf) # density matrix
            
        // elements for dm
        vhf = mf.get_veff(mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm, 0, None, 0, 0, 0)
        // Initial energy
        e_tot = mf.energy_tot(dm, h1e, vhf)
        logger.info(str("e_tot initial = "+str(e_tot)))
        energy = [e_tot]
        // commutator : residual
        r = commutator(fock,dm,s1e) # residual 

        //lists to save the iterates
        dmlist = [dm] 
        rlist = [r] # iterates of the current residual
        rlistIter = [] # the residuals family we keep at iteration k
        slist = [] # difference of residual (depending on the choice of CDIIS)
        rnormlist = [] # iterates of the current residual norm
        restartIt = [] # list of the iterations k when the R-CDIIS algorithm restarts
        mklist = [] # list of mk
        cnormlist = []
                        
        // init
        gamma = 1.0 
        mk = 0
        nbiter = 1
        // boolean to manage the first step
        notfirst = 0
        Restart = True // boolean to manage the QR decomposition when restart occurs

        // for the reader of the paper
        if(mode=="R-CDIIS"):
            tau=param
        elif(mode=="AD-CDIIS"):
            delta = param
        // while the residual is not small enough  
        while (np.linalg.norm(r[-1])>tol and nbiter<maxiter) {
            rlistIter.append(rlist) 
            rnormlist.append(np.linalg.norm(r))
            mklist.append(mk)
            logger.info("======================")
            logger.info("iteration: "+ str(nbiter))
            logger.info("mk value: "+str(mk))
            logger.info("||r(k)|| = "+str(np.linalg.norm(rlist[-1])))
            if(mk>0 and booldiis ) { // if there exist previous iterates and diis mode
                logger.info("size of Cs: "+str(np.shape(Cs)))
                if(mode=="R-CDIIS") {
                    if(modeQR=="full") {
                        if(mk==1 or Restart==True) { // if Q,R does not exist yet
                            Restart = False
                            Q,R = scipy.linalg.qr(Cs)
                        }
                        else { // update Q,R from previous one
                            Q,R = scipy.linalg.qr_insert(Q,R,Cs[:,-1],mk-1,'col')
                        }
                    }
                    else if(modeQR=="economic") { // modeQR="economic"
                        Q,R = scipy.linalg.qr(Cs,mode="economic")
                    }
                }
                else if(mode=="AD-CDIIS") {
                    Q,R = scipy.linalg.qr(Cs,mode="economic")
                }
                else if(mode=="FD-CDIIS") {
                    if(modeQR=="full") {
                        if(mk==1) { // if Q,R does not exist yet
                            Q,R = scipy.linalg.qr(Cs)
                        }
                        else if(mk<sizediis) { // we only add a column
                            Q,R = scipy.linalg.qr_insert(Q,R,Cs[:,-1],mk-1,'col')
                        }
                        else {
                            if(notfirst) { // if not the first time we reach the size
                                Q,R = scipy.linalg.qr_delete(Q,R,0,which='col') // we remove the first column
                            }
                            Q,R = scipy.linalg.qr_insert(Q,R,Cs[:,-1],mk-1,'col') // we add a column
                            notfirst = 1
                        }
                    }
                    else if(modeQR=="economic") { // modeQR="economic"
                        Q,R = scipy.linalg.qr(Cs,mode="economic")
                    }
                } 
                Q1 = Q[:,0:mk] // the orthonormal basis as the subpart of Q denoted Q1
                // solve the LS equation R1 gamma = -Q_1^T r^(k-mk) or -Q_1^T r^(k)
                // depending on the choice of algorithm, the RHS is not the same (last or oldest element)
                if(mode=="AD-CDIIS"  or mode=="FD-CDIIS") {
                    rhs = -np.dot(Q.T,np.reshape(rlist[-1],(-1,1))) // last : r^{k}
                }
                else if(mode=="R-CDIIS") {
                    rhs = -np.dot(Q.T,np.reshape(rlist[0],(-1,1))) // oldest : r^{k-m_k}
                }

                // compute gamma solution of R_1 gamma = RHS
                gamma = scipy.linalg.solve_triangular(R[0:mk,0:mk],rhs[0:mk],lower=False)
                // compute c_i coefficients
                c = np.zeros(mk+1)
                // the function gamma to c depends on the algorithm choice
                if(mode=="AD-CDIIS"  or mode=="FD-CDIIS") {
                    logger.info("size of c: "+str(np.shape(c)[0])+", size of gamma: "+str(np.shape(gamma)))
                    c[0] = -gamma[0] // c_0 = -gamma_1 (c_0, ... c_mk) and (gamma_1,...,gamma_mk)
                    for i in range(1,mk) { // 1... mk-1
                        c[i] = gamma[i-1]-gamma[i] // c_i=gamma_i-gamma_i+1
                    }
                    c[mk] = 1.0 - np.sum(c[0:mk])
                }
                else { // restart
                    c[0] = 1.0 - np.sum(gamma)
                    for i in range(1,mk+1) {
                        c[i] = gamma[i-1]
                    }
                }
                // dmtilde 
                dmtilde = 0.0*dm.copy() // init
                for i in range(mk+1) {
                    dmtilde = dmtilde+c[i]*dmlist[i]
                }
                cnormlist.append(np.linalg.norm(c,np.inf))
            }                                                                                
            else { // ROOTHAAN (if booldiis==False) or first iteration of cdiis
                dmtilde=dm.copy()
                cnormlist.append(1.0)
            }    
                                                                                                                    
            // computation of the new dm k+1 from dmtilde
            vhftilde = mf.get_veff(mol, dmtilde)
            focktilde = mf.get_fock(h1e, s1e, vhftilde, dmtilde, 0, None, 0, 0, 0)
            dm = Aufbau(focktilde,s1e,mf)
            vhf = mf.get_veff(mol, dm)
            fock = mf.get_fock(h1e, s1e, vhf, dm, 0, None, 0, 0, 0)
            dmlist.append(dm)
            e_tot = mf.energy_tot(dm, h1e, vhf)
            energy.append(e_tot)
            // residual
            r = commutator(fock,dm,s1e)
            logger.info("||r_{k+1}|| = "+str(np.linalg.norm(r)))
            // compute the s^k vector 
            if(mode=="AD-CDIIS" or mode=="FD-CDIIS") { //  as the difference between the r^{k+1} and the last r^{k}
                s = r-rlist[-1]
            }
            else if(mode=="R-CDIIS" ) { // as the difference between the r^k and the older r^{k-mk}
                s = r-rlist[0]
            }
            else if(mode=="Roothaan") {
                s = r.copy()
            }
            rlist.append(r)
            slist.append(s)
            if(mk==0 or not booldiis) { // we build the matrix of the s vector
                Cs = np.reshape(s,(-1,1))
            }
            else {
                Cs = np.hstack((Cs, np.reshape(s,(-1,1))))
            }                                                                 

            if(mode=="R-CDIIS") {
                if(mk>0) {              
                    if(tau*np.linalg.norm(Cs[:,-1])> np.linalg.norm(Cs[:,-1]-np.dot(Q1,np.dot(Q1.T,Cs[:,-1])))) {
                        restartIt.append(nbiter)
                        mk=minrestart-1
                        // reinitialization
                        Cs = Cs[:,-minrestart:]
                        slist = slist[-minrestart:]
                        rlist = rlist[-minrestart:]
                        dmlist = dmlist[-minrestart:]
                        restart = True
                    }
                    else {
                        mk = mk+1
                    }    
                }
                else { // if mk==0
                    mk = mk+1
                }
            }

            if(mode=="AD-CDIIS") { // mode Adaptive-Depth
                mk = mk +1
                outNbr = 0
                indexList = []
                for l in range(0,mk-1) {
                    if(np.linalg.norm(rlist[-1])<(delta*np.linalg.norm(rlist[l]))) {
                        outNbr = outNbr + 1
                        indexList.append(l)
                    }
                    else {
                        if(slidehole==False) {
                            break
                        }
                    }
                }
                if(indexList != []):
                    mk=mk-outNbr
                    logger.info("Indexes out :"+str(indexList))
                    // delete the corresponding s vectores
                    Cs = np.delete(Cs,indexList,axis=1)
                    for ll in sorted(indexList, reverse=True): { // delete elements of each lists
                        slist.pop(ll)
                        rlist.pop(ll)
                        dmlist.pop(ll)
                    }
            }
            else if(mode=="FD-CDIIS") { // keep only sizediis iterates
                if(mk == sizediis) {
                    logger.info(str(np.shape(Cs)))
                    Cs = Cs[:,1:mk+1]
                    logger.info(str(np.shape(Cs)))
                    dmlist.pop(0)
                    slist.pop(0)
                    rlist.pop(0)
                }
                if(mk<sizediis) {
                    mk = mk+1
                }
            }
            nbiter = nbiter +1
            dmlast = dm
            logger.info("e_tot = "+str(e_tot))
        }

        if(np.linalg.norm(r[-1])>tol and nbiter == maxiter) {
            conv = False
        }
        else {
            conv = True
        }
        return energy,conv,nbiter-1,rnormlist,mklist,cnormlist,dmlast

    }
}
*/
}
