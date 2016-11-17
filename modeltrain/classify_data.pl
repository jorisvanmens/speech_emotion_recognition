#!/usr/bin/perl
$do_standardise = 0; # TODO: implement speaker standardisation...
$do_select = 0;
$regression = 0;  # 0: classification model , 1: regression model

#--------------
use File::Basename;
require "arff-functions.pl";

if ($#ARGV < 0) {
  print "\nUsage: perl classify.pl corpus_path smile_extract_config arff_file model scale output\n\n";
  exit -1;
}

$xtract = "../../SMILExtract";

mkdir ("work");
$corp = $ARGV[0];
$arff_file = $ARGV[2];
$model = $ARGV[3];
$scale = $ARGV[4];
$output = $ARGV[5];
  $mode = "corp";
  unless (-d "$corp") {
    print "ERROR '$corp' is not a corpus directory or does not exist!\n";
    exit;
  }
  $corp =~ /\/([^\/]+)_FloEmoStdCls/;
  $cname = $1;
  $conf = $ARGV[1];
  unless ($conf) { $conf = "is09s.conf"; }
  $cb=$conf; $cb=~s/\.conf$//;
  $workpath = "work/$cname";
  $arff = "$workpath/$arff_file.arff";
  print " Corp: \"$corp\" \n";
  print " Conf: \"$conf\" \n";
  print " Cname: \"$cname\" \n";
  print " arff: \"$arff\" \n";

  mkdir("$workpath");


#extract features
if ($mode eq "corp") {
  print "-- Corpus mode --\n  Running feature extraction on corpus '$corp' ...\n";
  print "perl stddirectory_smileextract.pl \"$corp\" \"$conf\" \"$arff\ \n";
  system("perl stddirectory_smileextract.pl \"$corp\" \"$conf\" \"$arff\"");
}

# ? standardise features
if ($do_standardise) {
 print "NOTE: standardsise not implemented yet, svm-scale will do the job during building of model\n";
}

# ? select features
$lsvm=$arff; $lsvm=~s/\.arff$/.lsvm/i;
if ($do_select) {
 print "Selecting features (CFS)...\n";
 $fself = $arff;
 $fself=~s/\.arff/.fselection/i;
 system("perl fsel.pl $arff");
 $arff = "$arff.fsel.arff";
} else {
  $fself="";
  print "Converting arff to libsvm feature file (lsvm) ...\n";
  # convert to lsvm
  my $hr = &load_arff($arff);
  my $numattr = $#{$hr->{"attributes"}};
  if ($hr->{"attributes"}[0]{"name"} =~ /^name$/) {
    $hr->{"attributes"}[0]{"selected"} = 0;  # remove filename
  }
  if ($hr->{"attributes"}[0]{"name"} =~ /^filename$/) {
    $hr->{"attributes"}[0]{"selected"} = 0;  # remove filename
  }
  if ($hr->{"attributes"}[1]{"name"} =~ /^timestamp$/) {
    $hr->{"attributes"}[1]{"selected"} = 0;  # remove filename
  }
  if ($hr->{"attributes"}[1]{"name"} =~ /^frameIndex$/) {
    $hr->{"attributes"}[1]{"selected"} = 0;  # remove filename
    if ($hr->{"attributes"}[2]{"name"} =~ /^frameTime$/) {
      $hr->{"attributes"}[2]{"selected"} = 0;  # remove filename
    }
  }
  if ($hr->{"attributes"}[1]{"name"} =~ /^frameTime$/) {
    $hr->{"attributes"}[1]{"selected"} = 0;  # remove filename
  }
   #$hr->{"attributes"}[$numattr-1]{"selected"} = 0; # remove continuous label
  &save_arff_AttrSelected($arff,$hr);
  system("perl arffToLsvm.pl $arff $lsvm");
}

# classify model
print "Classifying using libsvm model ...\n";
system("perl classify_model.pl $lsvm $model $scale $output");


