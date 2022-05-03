### ::: Univariate CoxPH for feature selection, Chaudhary et al (2018) LIHC::: ###

library(survival)
library(survminer)
library(UCSCXenaTools)

cancer.type <- "BRCA"
model.type <- "DGCCA"
exp.name <- "exp2"
dir.create(paste0('./results/',cancer.type,'/',model.type,'/embeddings/',exp.name))
dir.create(paste0('./results/',cancer.type,'/',model.type,'/univcoxph/',exp.name))
dir.create(paste0('./results/',cancer.type,'/',model.type,'/clustering/',exp.name))

# embedding <-read.csv('./results/LIHC/DGCCA/embeddings/DGCCA_embeddings_exp26.csv',row.names = 1)
embedding <- read.csv(paste0('./results/',cancer.type,'/',model.type,'/embeddings/',exp.name,'/',model.type,'_embeddings.csv'), row.names = 1)
# embedding <- read.csv('./results/LIHC/DGCCA/embeddings/DGCCA_embeddings_exp25_epoch19_best2_9e-7.csv',row.names = 1)

#### survival
XenaGenerate(subset = XenaHostNames == "tcgaHub") %>%
  XenaFilter(filterCohorts = cancer.type) %>%
  XenaFilter(filterDatasets = "survival") -> df_todo

XenaQuery(df_todo) %>%
  XenaDownload(destdir = paste0("/projects/b1017/Jerry/cancer_subtyping/data/TCGA_",cancer.type,"/")) -> xe_download

surv <- XenaPrepare(xe_download)
if(is.list(surv) & (!is.data.frame(surv))){
  surv <- surv[[1]]
}
surv <- as.data.frame(surv)

#### univariate coxph归####分析
covariates <- colnames(embedding)
# merge survival
# surv <- surv[order(match(surv$sample, row.names(embedding))), ]
if(cancer.type == "LUSC"){
  surv <- surv[,c("xena_sample","OS.time","OS")]
  colnames(surv) <- c("sample", "OS.time", "OS")
}else{
  surv <- surv[,c("sample",'OS.time',"OS")]
}
row.names(surv) <- surv$sample; surv$sample <- NULL
embedding <- merge(embedding,surv,by='row.names',all.x=TRUE)
row.names(embedding) <- embedding$Row.names;embedding$Row.names <- NULL
colnames(embedding) <- c(covariates,'time','status')

# create formulas量，构建生存分析的公式
univ_formulas <- sapply(covariates,
                        function(x)
                          as.formula(paste('Surv(time, status)~', x)))

# coxph for every var归分析
univ_models <- lapply(univ_formulas, function(x) {
  model <- coxph(x, data = embedding)
  print(summary(model))
  model
})


#extract HR, 95% confidence interval and p-value值
univ_results <- lapply(univ_models,
                       function(x) {
                         x <- summary(x)
                         #获取p值
                         p.value <-
                           signif(x$logtest["pvalue"], digits = 3)
                         #获取HR
                         HR <- signif(x$coef[2], digits = 2)
                         
                         #获取95%置信区间
                         HR.confint.lower <-
                           signif(x$conf.int[, "lower .95"], 2)
                         HR.confint.upper <-
                           signif(x$conf.int[, "upper .95"], 2)
                         HR <- paste0(HR, " (",
                                      HR.confint.lower, "-", HR.confint.upper, ")")
                         res <- c(p.value, HR)
                         names(res) <-
                           c("p.value", "HR (95% CI for HR)")
                         return(res)
                       })

# make dataframe and transpose
res <- t(as.data.frame(univ_results, check.names = FALSE))
res <- as.data.frame(res)
write.table(file = paste0('./results/',cancer.type,'/',model.type,'/univcoxph/',exp.name,'/univariate_cox_result.txt'),
            res,
            quote = F,
            sep = "\t")


# pick significant vars
sig.vars <-
  res[which(as.numeric(as.character(res$p.value)) < 0.05), ] # TODO: benjamini-hochberg?
embedding.coxph <-
  embedding[, which(colnames(embedding) %in% row.names(sig.vars))]

# write to csv
write.csv(embedding.coxph,paste0('./results/',cancer.type,'/',model.type,'/embeddings/',exp.name,'/',model.type,'_embeddings_coxph.csv'))
# write.csv(embedding.coxph, './results/',cancer.type,'/AE/embeddings/AE_embeddings_coxph.csv')
