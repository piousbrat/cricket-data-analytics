#Download data: https://drive.google.com/open?id=1R4HIqNO3t25LU5uwfnaqb8rsDI0KN9tw
ball_info <- read.csv("~/ball_info.csv", stringsAsFactors=FALSE)
ball_info<-cbind(ball_info,score=rep(0,length(ball_info[,1])))
ball_info$score[which(ball_info$play_type=="run")]<-ball_info$score_value[which(ball_info$play_type=="run")]
ball_info$score[which(ball_info$play_type=="four")]<-4
ball_info$score[which(ball_info$play_type=="six")]<-6
wicket_value<-24.26656
ball_info$score[which(ball_info$play_type=="out")]<- -wicket_value


batsman.list <- unique(ball_info$batsman)
bowler.list <- unique(ball_info$bowler)
batsman.rating <- array(dim = length(batsman.list))
bowler.rating <- array(dim = length(bowler.list))
n=0

for(i in batsman.list){
  sum1=sum(ball_info$score[which(ball_info$batsman==i)])
  frq1=sum(ball_info$batsman==i)
  rat1=sum1*log(frq1, base = 120)
  batsman.rating[which(batsman.list==i)]=rat1
}

ball_info$score[which(ball_info$play_type=="wide")]<- 1
ball_info$score[which(ball_info$play_type=="no ball")]<- 1

for(i in bowler.list){
  sum1=sum(ball_info$score[which(ball_info$bowler==i)])
  frq1=sum(ball_info$bowler==i)
  rat1=sum1*log(frq1,base = 120)
  bowler.rating[which(bowler.list==i)]=rat1
}

batsman.rank=rank(-batsman.rating)
bowler.rank=rank(bowler.rating)

batsman_info <- as.data.frame(cbind(batsman=batsman.list,batrating=batsman.rating,batrank=batsman.rank))
batsman_info$batrating<-as.numeric(batsman.rating)
batsman_info$batrank<-as.numeric(batsman.rank)

bowler_info <- as.data.frame(cbind(bowler=bowler.list,ballrating=bowler.rating,ballrank=bowler.rank))
bowler_info$ballrating<- as.numeric(-bowler.rating)
bowler_info$ballrank<-as.numeric(bowler.rank)

batsman_info <- batsman_info[order(batsman_info$batrank),]
bowler_info <- bowler_info[order(bowler_info$ballrank),]

x_bat<-c(rep(1:361,each=5),rep(362,3))
x_ball<-c(rep(1:271,each=5),rep(272,4))
x_bat1<-c(rep(c(1,0,0,0,0),361),c(1,0,0))
x_bat2<-c(rep(c(0,1,0,0,0),361),c(0,1,0))
x_bat3<-c(rep(c(0,0,1,0,0),361),c(0,0,1))
x_bat4<-c(rep(c(0,0,0,1,0),361),c(0,0,0))
x_bat5<-c(rep(c(0,0,0,0,1),361),c(0,0,0))
x_ball1<-c(rep(c(1,0,0,0,0),271),c(1,0,0,0))
x_ball2<-c(rep(c(0,1,0,0,0),271),c(0,1,0,0))
x_ball3<-c(rep(c(0,0,1,0,0),271),c(0,0,1,0))
x_ball4<-c(rep(c(0,0,0,1,0),271),c(0,0,0,1))
x_ball5<-c(rep(c(0,0,0,0,1),271),c(0,0,0,0))

batsman_info <- cbind(batsman_info, x_bat, x_bat1, x_bat2, x_bat3, x_bat4, x_bat5)
bowler_info <- cbind(bowler_info, x_ball, x_ball1, x_ball2, x_ball3, x_ball4, x_ball5)


ball_info$play_type[which(ball_info$play_type=="out")] <- ball_info$wicket_how[which(ball_info$play_type=="out")]
ball_info$play_type[which(ball_info$is_keeper==TRUE & ball_info$play_type=="caught")] <- "caughtWK"
ball_info$play_type[which(ball_info$bowler==ball_info$fielder)]<-"caughtB"
ball_info$play_type[which(ball_info$play_type=="")] <- "no run"
ball_info$play_type[which(ball_info$play_type=="not out")]<-"no run"
legByeFreq<-table(ball_info$score_value[which(ball_info$play_type=="leg bye")])
legByeFreq<-legByeFreq/sum(legByeFreq)
bwWkFreq<-table(ball_info$score_value[which(ball_info$play_type=="run")])
bwWkFreq<-bwWkFreq/sum(bwWkFreq)
ballData<-as.data.frame(cbind(ball_info$batsman, ball_info$bowler, ball_info$play_type))
names(ballData)<-c("batsman","bowler","play_type")
c<-which(ballData$bowler=="" & ballData$batsman=="")
ballData2<-ballData[-c,]

y_cat <- unique(ballData$play_type)
y_mat <- as.data.frame(cbind(as.character(y_cat), diag(17)))
y_mat[,2:18]<-diag(17)
names(y_mat)<-c("play_type",as.character(unique(ballData$play_type)))

ballData3<-merge(x=ballData2, y=batsman_info)
ballData3<-merge(x=ballData3, y=bowler_info)
ballData3<-merge(x=ballData3, y=y_mat)
ref_data<-ballData3[,c(3,2,1,4:36)]
feed_data<-ballData3[,c(6:11, 14:36)]
feed_data$x_bat=feed_data$x_bat/max(feed_data$x_bat)
feed_data$x_ball=feed_data$x_ball/max(feed_data$x_ball)

#####NEURAL NETWORK
relu <- function(x){
  return(pmax(x,0))
}

reluprime <-function(x){
  x=x+0.000001
  return(pmax(x,0)/x)
}

sigmoid <- function(x){
  return(1/(1+exp(-x)))
}

sigmoidprime <- function(x){
  return(exp(-x)/(1+exp(-x))^2)
}

softmax <- function(x){
  x1=exp(x)
  n=length(x[,1])
  for(i in 1:n){
    x1[i,]=x1[i,]/sum(x1[i,])
  }
  return(x1)
}


trainIndex<-sample(1:529166, 370000)
trainData<-feed_data[trainIndex,]
testData<-feed_data[-trainIndex,]

Xtrain=as.matrix(trainData[,1:12])
Ytrain=as.matrix(trainData[,13:29])

nLayers=4
nUnits=array()
nUnits[1]=12
nUnits[2]=5
nUnits[3]=7
nUnits[4]=11
nUnits[5]=17
nUnits[6]=17

w=list()
a=list()
z=list()
dw=list()
da=list()
dz=list()

for(i in 1:nLayers){
  w[[i]]<-matrix(rnorm(nUnits[i]*nUnits[i+1]), nrow = nUnits[i])
  a[[i]]<-matrix(rep(0,1000*nUnits[i]), nrow = 1000)
  z[[i]]<-matrix(rep(0,1000*nUnits[i]), nrow = 1000)
  dw[[i]]<-matrix(rnorm(nUnits[i]*nUnits[i+1]), nrow = nUnits[i])
  da[[i]]<-matrix(rep(0,1000*nUnits[i]), nrow = 1000)
  dz[[i]]<-matrix(rep(0,1000*nUnits[i]), nrow = 1000)
}
a[[i+1]]<-matrix(rep(0,1000*nUnits[i+1]), nrow = 1000)
z[[i+1]]<-matrix(rep(0,1000*nUnits[i+1]), nrow = 1000)
da[[i+1]]<-matrix(rep(0,1000*nUnits[i+1]), nrow = 1000)
dz[[i+1]]<-matrix(rep(0,1000*nUnits[i+1]), nrow = 1000)

forwardprop <- function(X,w=w){
  a[[1]]=X
  w=w
  for(i in 1:nLayers){
    z[[i+1]]<-a[[i]]%*%w[[i]]
    a[[i+1]]<-relu(z[[i+1]])
  }
  a[[i+1]]<-softmax(z[[i+1]])
  return(as.list(c(a,z)))
}

backprop <- function(X,a,w,z,lRate){
  dz[[nLayers+1]]<-X
  a=a
  #print(length(a[[2]]))
  w=w
  lRate=lRate
  z=z
  for(i in nLayers:1){
    dw[[i]]<-(t(a[[i]])%*%dz[[i+1]])/1000
    da[[i]]<-dz[[i+1]]%*%t(w[[i]])
    w[[i]]<-w[[i]]-lRate*dw[[i]]
    dz[[i]]<-da[[i]]*reluprime(z[[i]])
  }
  return(w)
}

##training

lRate=0.005
cost=1000
cost1=cost+1
epoch=0
while(abs(cost-cost1)>0.01){
  epoch=epoch+1
  print(paste("Epoch#",epoch,sep = ""))
  for(i in 1:370){
    cost1<-cost
    X=Xtrain[((i-1)*1000+1):(i*1000),]
    Y=Ytrain[((i-1)*1000+1):(i*1000),]
    #a[[1]] <- X
    az<-forwardprop(X,w)
    for(j in 1:(nLayers+1)){
      a[[j]]<-az[[j]]
      z[[j]]<-az[[j+nLayers+1]]
    }
    cost<<- -sum(Y*log(a[[nLayers+1]]))/1000
    if(is.nan(cost)){break()}
    dz[[nLayers+1]]<-a[[nLayers+1]]-Y #init back
    w<-backprop(dz[[nLayers+1]],a,w,z,lRate)
    #print(w)
    print(paste("Batch:",i,", Cost:", cost))
  }
}

##testing

Xtest=as.matrix(testData[,1:12])
Ytest=as.matrix(testData[,13:29])
X=Xtest
#a[[1]]=X
a<-forwardprop(X,w)
Yhat<-a[[5]]
cost= -sum(Ytest*log(Yhat))/length(Ytest[,1])
cost
playtype<-colnames(Y)

##predictor
lastout=1
overPredictor <- function(batsmanOnX,batsmanOffX,bowlerX){
  X1<-c(as.numeric(batsman_info[grep(batsmanOnX, batsman_info$batsman),4:9]),
       as.numeric(bowler_info[grep(bowlerX, bowler_info$bowler),4:9]))
  X1[1]<-X1[1]/362
  X1[7]<-X1[7]/272
  X2<-c(as.numeric(batsman_info[grep(batsmanOffX, batsman_info$batsman),4:9]),
        as.numeric(bowler_info[grep(bowlerX, bowler_info$bowler),4:9]))
  X2[1]<-X2[1]/362
  X2[7]<-X2[7]/272
  X=rbind(X1,X2)
  for(i in 1:(nLayers+1)){
    a[[i]]<-matrix(rep(0,2*nUnits[i]), nrow = 2)
    z[[i]]<-matrix(rep(0,2*nUnits[i]), nrow = 2)
  }
  a<-forwardprop(X,w)
  Yhat1<-as.numeric(a[[5]][1,])
  Yhat2<-as.numeric(a[[5]][2,])
  lastout<<-lastout%%6
  i=lastout
  while(i<=6){
    sum1=0
    sum2=0
    shots=runif(1)#rnorm(1,0.5,0.13)
    for(j in 1:length(Yhat1)){
      sum1=sum1+Yhat1[j]
      sum2=sum2+Yhat2[j]
      if(shots<sum1){
        if(playtype[j]=="run"){
          run=runif(1)
          sumrun=0
          irun=1
          while(run>sumrun){
            sumrun<-sumrun+bwWkFreq[irun]
            irun=irun+1
          }
          print(paste("Batsman on strike:",batsmanOnX,irun-1,"run(s)",sep = " "))
          if(irun%%2!=1){
            tmp=batsmanOnX
            batsmanOnX=batsmanOffX
            batsmanOffX=tmp
            
            tmp=Yhat1
            Yhat1=Yhat2
            Yhat2=tmp
            
            tmp=sum1
            sum1=sum2
            sum2=tmp
          }
        }
        else if(playtype[j]=="leg bye" | playtype[j]=="bye"){
          run=runif(1)
          sumbye=0
          ibye=1
          while(run>sumbye){
            sumbye<-sumbye+legByeFreq[ibye]
            ibye=ibye+1
          }
          print(paste("Batsman on strike:",batsmanOnX,ibye-1,"(leg)bye run(s)",sep = " "))
          if(ibye%%2!=1){
            tmp=batsmanOnX
            batsmanOnX=batsmanOffX
            batsmanOffX=tmp
            
            tmp=Yhat1
            Yhat1=Yhat2
            Yhat2=tmp
            
            tmp=sum1
            sum1=sum2
            sum2=tmp
          }
        }
        else{
          print(paste("Batsman on strike:",batsmanOnX,playtype[j],sep = " "))
        }
        break()
      }
    }
    i=i+1
    if(j %in% c(3,15)){i=i-1}
    lastout<<-i
    if(j %in% c(7,9:14,16,17)){break()}
  }
}

overPredictor("Gayle","Dhoni","Narine")
#>overPredictor("Gayle","Dhoni","Narine")
#[1] "Batsman on strike: Gayle 1 (leg)bye run(s)"
#[1] "Batsman on strike: Dhoni 1 run(s)"
#[1] "Batsman on strike: Gayle 1 run(s)"
#[1] "Batsman on strike: Dhoni no run"
#[1] "Batsman on strike: Dhoni six"
#[1] "Batsman on strike: Dhoni 1 run(s)"