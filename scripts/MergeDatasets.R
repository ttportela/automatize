library(data.table)
library(dplyr)

args <- commandArgs(trailingOnly = TRUE)

if (length(args)<1) {
  stop("Please, DIR_PATH of csv files must be supplied", call.=FALSE)
}

dir <- args[1] 
#dir <- "D:/Users/andres/kbs2018/results/KBS2018/E4/gowalla6/Run/Movelets/gowalla6b_ED/"


files.train <- list.files(dir,recursive = T, pattern = "train.csv")

files.test <- list.files(dir,recursive = T, pattern = "test.csv")

print(files.train)

print("Loading training files")
train.dt <- bind_cols( lapply(files.train[-length(files.train)], function(file.name){

  file.dt <- fread( file.path(dir,file.name) )

  file.dt
}))

column.index <- c( grep( "class", colnames(train.dt), invert = T), grep( "^class$", colnames(train.dt)) )
train.dt1 <- data.frame(train.dt)[,column.index]

print("Writing train file")
fwrite(train.dt1, file = paste0(dir,"/","train.csv"), row.names = F)


print("Loading testing files")
test.dt <- bind_cols( lapply(files.test, function(file.name){
  
  file.dt <- fread( file.path(dir,file.name) )
  
  file.dt
}))

print("Writing test file")
column.index <- c( grep( "class", colnames(test.dt), invert = T), grep( "^class$", colnames(test.dt)) ) # By Tarlis
test.dt1 <- data.frame(test.dt)[, column.index]

fwrite(test.dt1, file = paste0(dir,"/","test.csv"), row.names = F)

print("Done.")
