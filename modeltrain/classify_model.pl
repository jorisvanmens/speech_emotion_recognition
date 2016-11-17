#!/usr/bin/perl

if ($#ARGV < 0) {
  print "\nUsage: perl classify_model.pl lsvm model scale output\n\n";
  exit -1;
}
$input_lsvm = $ARGV[0];
$model = $ARGV[1];
$scale = $ARGV[2];
$output = $ARGV[3];

$scaled_lsvm = $input_lsvm; $scaled_lsvm =~ s/\.lsvm/.scaled.lsvm/;
 
# scale features, build model

print "Scale: $scale\n";
system("libsvm-small/svm-scale -r $scale $input_lsvm > $scaled_lsvm");

print "  classifying using model...\n";
#classification:
system("libsvm-small/svm-predict -b 1 $scaled_lsvm $model $output");

