#!/usr/bin/perl

#--------------
use File::Basename;
require "arff-functions.pl";

if ($#ARGV < 0) {
  print "\nUsage: perl arff_to_lsvm.pl arff_file\n\n";
  exit -1;
}

$arff=$ARGV[0];
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


