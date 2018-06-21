#!/usr/bin/perl
# Created by AV Sivaprasad on Jun 07, 2018
# Last modified by AV Sivaprasad on Jun 07, 2018
# This program can be used under the GNU public licence conditions.
# ------------------------------------------------------------------------------

$file = $ARGV[0];
$format = $ARGV[1];
#print "format = $format";
if (!$file)
{
	print "Usage: /g/data/u46/users/sa9525/avs/WOfS/get_lat_lon.pl filename\n"; exit;
}
my $gdalinfo = `gdalinfo NETCDF:$file:slope | grep NC_GLOBAL`;
@gdalinfo = split (/\n/, $gdalinfo);
foreach $line (@gdalinfo)
{
	if ($line =~ /geospatial_lat_max=(.*$)/) { $lat_max = $1; }
	if ($line =~ /geospatial_lat_min=(.*$)/) { $lat_min = $1; }
	if ($line =~ /geospatial_lon_max=(.*$)/) { $lon_max = $1; }
	if ($line =~ /geospatial_lon_min=(.*$)/) { $lon_min = $1; }
}
if ($format eq "remote")
{
	print "$lat_min $lat_max $lon_min $lon_max";
}
else
{
	$query = "query = \{
           'lat': ($lat_min, $lat_max),
           'lon': ($lon_min, $lon_max)
         \}";
	print "$query\n";        
}
    
