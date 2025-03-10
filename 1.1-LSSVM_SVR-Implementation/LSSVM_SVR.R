#===========================================
# Statistical Computing - Project 1
# Implementation of Machine Learning Methods
#===========================================

#===========================================
# Problem 1 - Part 1: LSSVM Implementation
#===========================================

# Load and preprocess data
data <- read.table("./pb2.txt")

#spliting the dataset into training and test
require(caTools)
set.seed(351969) 
n=nrow(data)
for (i in 1:n){
  if (data[i,1] > 1){
    data[i,1]<--1
  }
}
sample = sample.split(data[,1], SplitRatio = .75)
train_data = subset(data, sample == TRUE)
test_data  = subset(data, sample == FALSE)


# Separating Y
Y <- train_data[,1]
# Separating Xs
X <- as.matrix(train_data[,2:5],ncol=4)
N <- length(Y)


#-------------------------------------------
# LSSVM Core Functions
#-------------------------------------------

# Gaussian RBF kernel implementation
rbf_kernel <- function(x1,x2,gamma){
  K<-exp(-(1/gamma^2)*t(x1-x2)%*%(x1-x2))
  return(K)
}


gamma<-2.5
rbf_kernel(X[1,],X[2,],gamma)

# Main LSSVM training function
LSSVMtrain <- function(X,Y,C=Inf, gamma=2,esp=1e-10){
  
  
  H_matrix <- function(X,Y,C=Inf, gamma=1.5,esp=1e-10){
    
    N<-length(Y)
    H<-matrix(0,N,N)
    X<-as.matrix(X);Y<-as.vector(Y)
    
    for(i in 1:N){
      for(j in 1:N){
        H[i,j]<-Y[i]*Y[j]*rbf_kernel(X[i,],X[j,],gamma)
      }
    }
    # Dm= Dmat; matrix appearing in the quadratic function to be minimized.
    H<-H+diag(N)*1e-12 # adding a very small number to the diag, some trick
    return(H)
  }
  
  H <- H_matrix(X,Y,C=12,gamma=13)
  
  #Verifying whether H is a positive definite matrix or not
  library(matrixcalc)
  is.positive.definite(H)
  
  
  #-------------------------------------------
  # Conjugate Gradient Implementation
  #-------------------------------------------
  # Solves linear system Ax = b iteratively
  ConjugateGrad <- function(A = NULL, B = NULL, x0 = NULL, i = 1500, thresh = 0.00001){
    
    r0 <- B
    p_1<- r0
    
    for(i in 1:i){
      
      lambda_i <- sum(r0*r0) / as.numeric((t(p_1) %*% A) %*% p_1)
      x_i <- x0 + (lambda_i * p_1)
      r_i <- r0 - lambda_i *  A %*% p_1
      
      beta_i <- sum(r_i * r_i) / sum(r0*r0)
      p_i <- r_i + beta_i * p_1
      
      x0 <- x_i
      p_1 <- as.numeric(p_i) 
      r0 <- r_i
      
      
      
      if(norm(r0) <= thresh){
        break
      }
      
    }
    
    return(x_i)
    
  }
  
  
  
  d2 <- rep(1,N)
  x0 <- rep(0, N)
  
  
  
  eta<- ConjugateGrad(H, Y, x0)
  # 949 iteration
  
  
  v<- ConjugateGrad(H, d2, x0)
  # 1087 iteration
  
  # Step 2
  s <- sum(t(Y)*eta)
  
  #Step 3
  b <- sum(t(eta)*d2)/s
  
  
  alpha_org <- v - eta*b
  indx<-which(alpha_org != 0,arr.ind=TRUE)
  alpha<-alpha_org[indx]
  nSV<-length(indx)
  if(nSV==0){
    throw("no solution of alpha for these data points")
  }
  Xv<-X[indx,]
  Yv<-Y[indx]
  Yv<-as.vector(Yv)
  
  
  list(indx=indx,alpha_org=alpha_org,alpha=alpha, H=H, b=b, nSV=nSV, Xv=Xv, Yv=Yv, gamma=gamma)
}

#-------------------------------------------
# Model Training and Evaluation
#-------------------------------------------
# Train LSSVM model
LSSVM_Train_model <- LSSVMtrain(X,Y,C=12,gamma=2) 
LSSVM_Train_model$alpha



### Predict the class of an object X
# Test data
Test_Y <- test_data[,1]
# Separating Xs
Test_X <- as.matrix(test_data[,2:5],ncol=4)
n <- nrow(Test_X)

# Prediction function for new data points
LSSVMpredict <- function(x,model){
  alpha<-model$alpha
  b<-model$b
  Yv<-model$Yv
  Xv<-model$Xv
  nSV<-model$nSV
  gamma<-model$gamma
  ayK <- numeric(nSV)
  Y_hat <- numeric(n)
  for (m in 1:n){
    for(i in 1:nSV){
      ayK[i]<-alpha[i]*Yv[i]*rbf_kernel(Xv[i,],x[m,],gamma)
    }
    Y_hat[m] <- sign(sum(ayK)+b)
  }
  return(Y_hat)
}


Pred_LSSVM <- LSSVMpredict(Test_X,LSSVM_Train_model);Pred_LSSVM

#Error rate:

table(Pred_LSSVM, Test_Y)
mean(Pred_LSSVM != Test_Y)

############################## End of part 1 #################################


#===========================================
# Problem 1 - Part 2: QR Updates
#===========================================

# New raw update

#Call data
data <- read.table("./pb2.txt")

Data <- as.matrix(data, ncol=5)
# Separating Y
Y <- Data[,1]
# Separating Xs
X <- as.matrix(Data[,2:5],ncol=4)
N <- length(Y)
#Recoding Y as 1 and -1
for (i in 1:N){
  if (Y[i] > 1){
    Y[i]<--1
  }
}



require('quadprog')

## Defining the Gaussian kernel
rbf_kernel <- function(x1,x2,gamma){
  K<-exp(-(1/gamma^2)*t(x1-x2)%*%(x1-x2))
  return(K)
}


gamma<-2
rbf_kernel(X[1,],X[2,],gamma)


k <- function(X,Y, gamma=1.5,esp=1e-10){
  
  N<-length(Y)
  H<-matrix(0,N,N)
  X<-as.matrix(X);Y<-as.vector(Y)
  
  for(i in 1:N){
    for(j in 1:N){
      H[i,j]<-Y[i]*Y[j]*rbf_kernel(X[i,],X[j,],gamma)
    }
  }
  
  # Dm= Dmat; matrix appearing in the quadratic function to be minimized.
  H<-H+diag(N)*1e-12 # adding a very small number to the diag, some trick
}

H <- k(X,Y,gamma=2)

#Veryfying whether H is a positive definite matrix or not
library(matrixcalc)
is.positive.definite(H)


#-------------------------------------------
# Givens Rotation Implementation
#-------------------------------------------
# Algorithm 1.1: Compute Givens rotation parameters
givens <- function(a,b) {
  if (b == 0){ 
    c=1
    s=0
  } else if (abs(b) >= abs(a)) {
    t = -a/b
    s = 1/sqrt(1+t**2)
    c = s*t
  } else {
    t = -b/a
    c = 1/sqrt(1+t**2)
    s = c*t
  }
  list(c=c,s=s)
}




## Find K

row=c(0,rep(1,N))
col = rep(1,N)
K = cbind(col,H)
K1 = as.matrix(rbind(row,K))


### New raw for K
x_raw = c(15, 21, 26, 21)
y_raw = 1
M = length(y_raw)
Ur<-numeric(0)


N<-length(Y)

for(j in 1:N){
  Ur[j]<-y_raw*Y[j]*rbf_kernel(x_raw,X[j,],gamma)
}

Ur <- c(1,Ur)


#-------------------------------------------
# Matrix Update Functions
#-------------------------------------------
# Update R matrix using Givens rotations (Algorithm 2.6)

# Get QR of K
Q = qr.Q(qr(K1))
R = qr.R(qr(K1))
b = c(0, rep(1, N))

#  Algorithm 2.6 to update R
c <- numeric(length(Ur))
s <- numeric(length(Ur))
d = t(Q)%*%b
mu = y_raw 


for (j in 1:length(Ur)){
  
  
  c[j] = givens(R[j,j], Ur[j])$c
  s[j] = givens(R[j,j], Ur[j])$s
  R[j,j] = c[j]*R[j,j] - s[j]*Ur[j]
  
  #updatd the jth row of R and u 
  
  i = 1
  while(i != length(Ur)){
    i = i+1
    t1 = R[j,i]
    t2 = Ur[i]
    R[j, i] = c[j]*t1 - s[j]*t2
    Ur[i] = s[j]*t1 + c[j]*t2
  }
  
  #Update jth row of d and mu 
  t1 = d[j]
  t2 = mu
  d[j] = c[j]*t1 - s[j]*t2
  mu = as.numeric(s[j]*t1 + c[j]*t2)
  
} # End of For Loop 



R1 = rbind(R, 0)
d2 = c(d, mu)

#########
temp_row <- rep(0,ncol(Q))
Q1 <- rbind(Q,temp_row)
temp_col <- c(temp_row,1)
Q2 <- cbind(Q1,temp_col)


# Update Q matrix (Algorithm 2.7)
m=ncol(Q1)
for (j in 1:length(Ur)){
  
  t1 = Q2[1:m+1,j]
  t2 = Q2[1:m,m+1]
  Q2[1:m+1,j] = c[j] * t1 - s[j] *t2
  Q2[1:m,m+1] = s[j] * t1 - c[j] * t2
  
}

# updated Q
Q2


### Updating column ###

ui <- c(Ur,1)
u = t(Q2) %*% ui
R_tilda <- cbind(R1,u)

############################## End of part 2 ################################


#===========================================
# Problem 1 - Part 3: Incremental Method
#===========================================

# Incremental


H <- LSSVM_Train_model$H

R<-qr.R(qr(H))
Q<-qr.Q(qr(H))
n <- nrow(R)
y <- c(0,rep(1,(n-1)))
b <- t(Q)%*%y


# Solve triangular system incrementally
incremental=function(R,b)
{ 
  if(!is.matrix(R)) stop("R must be a matrix")
  D=dim(R)
   
  p=D[2]
  b=as.matrix(b)
  
  for (j in seq(p,1,-1))
  {  
    b[j,1]=b[j,1]/R[j,j]
    if((j-1)>0)
      b[1:(j-1),1]=b[1:(j-1),1]-(b[j,1]*R[1:(j-1),j])
  }  
  return(b)
}

incremental(R,b)


############################## End of Problem 1, part 3 ################################


#===========================================
# Problem 2: Support Vector Regression
#===========================================

# the data
library(MASS)
library(quadprog)
Boston

#spliting the dataset into training and test
require(caTools)
set.seed(351969) 
sample = sample.split(Boston[,14], SplitRatio = .75)
train_data = subset(Boston, sample == TRUE)
test_data  = subset(Boston, sample == FALSE)

# Separating Y
Y <- train_data[,14]
# Separating Xs
X <- as.matrix(train_data[,1:13],ncol=13)
N <- length(Y)


# Kernel function

## Defining the Gaussian kernel
rbf_kernel <- function(x1,x2,gamma){
  kernel<-exp(-(1/gamma^2)*t(x1-x2)%*%(x1-x2))
  return(kernel)
}

gamma<-2


#-------------------------------------------
# SVR Implementation
#-------------------------------------------
# Train SVR model with quadratic programming
SVRtrain <- function(X,Y,C=Inf, gamma=2,esp=1e-10){
  
  if(!is.numeric(gamma) || gamma <= 0) stop("gamma must be positive")
  X<-as.matrix(X);Y<-as.vector(Y)
  
  
  k<-matrix(0,N,N)
  for(i in 1:N){
    for(j in 1:N){
      k[i,j]<-rbf_kernel(X[i,],X[j,],gamma)
    }
  }
  
  
  # k= Dmat; matrix appearing in the quadratic function to be minimized.
  K<-rbind(cbind(k,-k),(cbind(-k,k)))
  K<-K+(diag(2*N)*1e-7)
  T <- rbind(cbind(diag(N),-diag(N)),(cbind(diag(N),-diag(N))))
  H <- t(T) %*% K %*% T
  H<-H+(diag(2*N)*1e-7) # adding a very small number to the diag, some trick
  
  
  # dv =  dvec = p = vector appearing in the quadratic function to be minimized.
  dv = t(as.vector(cbind((esp - Y),(esp + Y))))
  
  # meq= meq; the first meq constraints are treated as equality constraints, all further as inequality constraints (defaults to 0).
  meq<-1
  
  #Am = Amat;	matrix defining the constraints under which we want to minimize the quadratic function.
  Am<-cbind(matrix(c(rep(1,N),rep(-1,N))),diag(2*N))
  
  #bv = bvec;	vector holding the values of b_0 (defaults to zero).
  bv<-rep(0,1+2*N) # the 1 is for the sum(alpha)==0, others for each alpha_i >= 0
  
  if(C!=Inf){
    # an upper bound is given
    Am<-cbind(Am,-1*diag(2*N))
    bv<-c(cbind(matrix(bv,1),matrix(rep(-C,2*N),1)))
  }
  
  alpha_org<-solve.QP(Dmat=H,dvec = dv,Amat = Am,meq=meq,bvec=bv)$solution
  alp_i <- alpha_org[1:N]
  alp_i_star <- alpha_org[(N+1):(2*N)]
  alp_constraint <-alp_i*alp_i_star
  indx2<-which(alp_constraint>esp,arr.ind=TRUE)
  alpha_i <- alp_i[indx2]
  alpha_i_star <- alp_i_star[indx2]
  
  nSV<-length(indx2)
  if(nSV==0){
    stop("QP is not able to give a solution for these data points")
  }
  
  Xv<-X[indx2,]
  Yv<-Y[indx2]
  Yv<-as.vector(Yv)
  
  b <- numeric(nSV)
  wx <- numeric(nSV)
  for (i in 1:nSV){
    for (m in 1:nSV){
      wx[m] <- (alpha_i[m]-alpha_i_star[m])*rbf_kernel(Xv[m,],Xv[i,],gamma)
    }
    b[i]<-Yv[i]-sum(wx)
    
  }
  w0 <- mean(b)
  
  list(alpha_org=alpha_org,alpha_i=alpha_i,alpha_i_star=alpha_i_star,b=b, w0=w0, nSV=nSV, Xv=Xv, Yv=Yv, gamma=gamma)
}

Training_model<-SVRtrain(X,Y,C=15,gamma=2)


### Predict the class of an object X
Test_X <- as.matrix(test_data[,1:13],ncol=13)
Test_Y <- test_data[,14]
n <- nrow(Test_X) 

# Predict using trained SVR model
SVR_predict <- function(x,model){
  alpha_i<-model$alpha_i
  alpha_i_star<-model$alpha_i_star
  b<-model$w0
  Yv<-model$Yv
  Xv<-model$Xv
  nSV<-model$nSV
  gamma<-model$gamma
  wx <- numeric(nSV)
  Y_hat <- numeric(n)
  for (i in 1:n){
    for (m in 1:nSV){
      wx[m] <- (alpha_i[m]-alpha_i_star[m])*rbf_kernel(Xv[m,],x[i,],gamma)
    }
    
    Y_hat[i] <- sum(wx) + b
  }
  
  return(Y_hat)
}


Predicted_Y <- SVR_predict(Test_X,Training_model);Predicted_Y
Y_hat <- t(as.data.frame(Predicted_Y))
MSE <- (sum((Y_hat - Test_Y)**2))/n
MSE
