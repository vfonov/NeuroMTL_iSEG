library(tidyverse)
library(zoo)
library(grid)
library(stringr)

theme_set(theme_bw(base_size = 14, base_family = "Arial"))

rm(list=ls())

fnames=c('iseg_train/v2_s0_1_fold_progress.txt','iseg_train/v2_s3_1_fold_progress.txt')
titles=c('ACE-IBIS training','iSEG training')


prog<-vector(mode = "list", length = length(fnames))
for(i in seq(1,2)) {
    fname=fnames[i]
    prog[[i]]=read_csv(fname)
    prog[[i]]$title=titles[i]
}

progress<-bind_rows(prog)

# remove v_kappa where it's undefined
#progress<-progress %>% mutate( v_kappa = ifelse(batch>19, v_kappa, NA) )

bw=20
progress<-progress %>% filter(title=='ACE-IBIS training'| batch>19) %>% group_by(title) %>% arrange(batch) %>% 
   mutate(rv_kappa=rollmean(x = v_kappa, bw, fill='extend'),
           r_kappa=rollmean(x = kappa,   bw, fill='extend'),
           r_error=rollmean(x = error,   bw, fill='extend')
   )
   
png('iSEG_train_progress.png',width=600,height=500)
grid.newpage()

p1<-ggplot(data=progress,aes(x=batch,y=kappa))+
    theme(
    legend.text = element_text(face = 'bold', vjust = 0.2, size = 10),
    axis.text   = element_text(face = 'bold', vjust = 0.2, size = 10),
    strip.text  = element_text(face = 'bold', vjust = 0.2, size = 10),
    axis.title  = element_text(face = 'bold', vjust = 0.2, size = 10),
    plot.margin = unit(c(0.2,0.2,0.2,0.2), "cm"),
    legend.position="bottom",
    legend.title=element_blank()
    )+
    geom_line(aes(y=rv_kappa,col='Validation'),alpha=1.0,size=1)+
    geom_line(aes(y=r_kappa,col='Training'),alpha=1.0,size=1)+
    facet_wrap(~title,ncol=2, scales="free_x")+
    ylab('')+xlab('minibatch')+
    ggtitle("Generalized kappa overlap")+
    scale_color_discrete(breaks=c("Validation","Training"))
    
p2<-ggplot(data=progress,aes(x=batch,y=error))+
    theme(
    axis.text  = element_text(face = 'bold', vjust = 0.2, size = 10),
    strip.text = element_text(face = 'bold', vjust = 0.2, size = 10),
    axis.title = element_text(face = 'bold', vjust = 0.2, size = 10),
    plot.margin = unit(c(0.2,0.2,0.2,0.2), "cm")
    )+
    geom_line(aes(y=r_error,col='Training'),alpha=1.0,size=2, show.legend=FALSE)+
    facet_wrap(~title,ncol=2, scales="free_x")+
    ggtitle("-log10(metric)")+
    ylab('')+xlab('')+
    scale_color_discrete(breaks=c("Validation","Training"))


pushViewport(viewport(layout = grid.layout(2, 1)))
print(p2, vp = viewport(layout.pos.row = 1,layout.pos.col = 1))
print(p1, vp = viewport(layout.pos.row = 2,layout.pos.col = 1))



# 
l=10
prefix='iseg_train/v2_s3'
# 
val<- vector(mode = "list", length = l)

for(i in seq(1,l)) {
    fname=paste(prefix,i,'fold_similarity.csv',sep='_')
    print(fname)
    val[[i]]=read.csv(fname)
}

cvv<-bind_rows(val)

# remove flipped scans
cvv<-cvv %>% filter(!str_detect(subject,'_f.mnc')) %>% rename(CSF=kappa_nil,GM=kappa_nil.1,WM=kappa_nil.2)

cvv_=cvv
cvv_$error<-NULL
cvv_$fold<-NULL
cvv_$subject<-NULL
cvv_$gen_kappa<-NULL

cvvk<-stack(cvv_)
names(cvvk)=c('Kappa','Structure')
cvvk$Structure_S<-as.factor(cvvk$Structure)
med<-cvvk %>% group_by(Structure_S) %>% summarize(mean_kappa=mean(Kappa),sd_kappa=sd(Kappa))
#med<-med %>% mutate(text<-sprintf('%0.3f',mean_kappa))

print(med)

png('iSEG_train_cvv_result.png',width=500,height=400)
ggplot(data=cvvk,aes(y=Kappa,x=Structure_S))+
    theme(
    axis.text  = element_text(face = 'bold', vjust = 0.2, size = 12),
    strip.text = element_text(face = 'bold', vjust = 0.2, size = 12),
    axis.title = element_text(face = 'bold', vjust = 0.2, size = 12),
    plot.margin = unit(c(0.2,0.2,0.2,0.2), "cm")
    )+
    geom_boxplot()+xlab('')+ylab('dice kappa')+
    theme_bw()

#write.csv(cvvk,file=paste(out_prefix,'cvv_ind.csv',sep='_'),quote=F)


