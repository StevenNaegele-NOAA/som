#!/bin/bash

myyears=("2015") ### "2016" "2017" "2018" "2019" "2020" "2021" "2022" "2023")
mymonths=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12")
mydays28=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28")
mydays29=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28" "29")
mydays30=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28" "29" "30")
mydays31=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24" "25" "26" "27" "28" "29" "30" "31")

##for myyear in ${myyears[@]};
##do
##    for mymonth in ${mymonths[@]};
##    do
##	if [ $mymonth == "01" ] || [ $mymonth == "03" ] || [ $mymonth == "05" ] || [ $mymonth == "07" ] || [ $mymonth == "08" ] || [ $mymonth == "10" ] || [ $mymonth == "12" ]; then
##	    for myday in ${mydays31[@]};
##	    do
##                hsi get -B /BMC/fdr/Permanent/$myyear/$mymonth/$myday/data/radar/mrms/20*.zip
##	    done
##        elif [ $mymonth == "02" ] && [ $myyear != "2016" ] && [ $myyear != "2020" ]; then
##	    for myday in ${mydays28[@]};
##            do
##                hsi get -B /BMC/fdr/Permanent/$myyear/$mymonth/$myday/data/radar/mrms/20*.zip
##            done
##        elif [ $mymonth == "02" ] && ([$myyear == "2016"] || [$myyear == "2020" ]); then
##	    for myday in ${mydays29[@]};
##            do
##                hsi get -B /BMC/fdr/Permanent/$myyear/$mymonth/$myday/data/radar/mrms/20*.zip
##	    done
##	elif [ $mymonth == "04" ] or [ $mymonth == "06" ] or [ $mymonth == "09" ] or [ $mymonth == "11"]; then
##	    for myday in ${mydays[@]};
##            do
##                hsi get -B /BMC/fdr/Permanent/$myyear/$mymonth/$myday/data/radar/mrms/20*.zip
##            done
##        fi
##    done
##    hsi ls -l /BMC/fdr/Permanent/$myyear/*/*/data/radar/mrms/20*.zip
##done
hsi get -B /BMC/fdr/Permanent/2015/01/*/data/radar/mrms/20*.zip

mypath=/scratch/BMC/wrfruc/naegele/MRMS_HPSS_data

for myfile in "$mypath"/*00.zip
do
    unzip $myfile
done

for myfile in "$mypath"/*Gauge.zip
do
    unzip $myfile
done

