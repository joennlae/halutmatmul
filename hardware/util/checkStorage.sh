#!/bin/bash

TORTIN=`echo tortin{1,2,3,4,5,6,7,8,9,10,11,12}`
DAHU=`echo dahu{1,2,3,4,5,6,7,8,9,10,11,12}`
ARINA=`echo arina{1,2,3,4,5,6,7,8,9,10}`
ATTELAS=`echo attelas{1,2,3,4,5,6,7,8}`
PISOC=`echo pisoc{1,2,3,4,5,6,7,8,9,10}`
MONTFORT0=`echo mont-fort{1,2,3,4,5,6,7,8,9}`
MONTFORT1=`echo mont-fort1{0,1,2,3,4,5,6,7,8,9}`
MONTFORT2=`echo mont-fort2{0,1,2,3,4,5,6,7,8,9}`
MONTFORT3=`echo mont-fort3{0,1,2,3,4,5,6,7,8}`
MONTFORT10=`echo mont-fort{101,102}`
MONTFORT="$MONTFORT0 $MONTFORT1 $MONTFORT2 $MONTFORT3 $MONTFORT10"
FENGA=`echo fenga{1,2,3,4,5,6,7,8,9}`
LARAIN=`echo larain{1,2,3,4,5,6,7,8,9}`
LAGREV=`echo lagrev{1,2,3,4,5}`
SASSAUNA=`echo sassauna{1,2,3,4}`
VILAN=`echo vilan{1,2}`
DOLENT=`echo dolent{1,2,3,4}`
OJOS=`echo ojos{1,2,3,4,5,6}`

scratch_folders=("scratch" "scratch2" "scratch3")
DAYS=365
SERVERS_OPTION=$PISOC

_INFO_OPTION=0
_CHECK_OPTION=0
_DELETE_OPTION=0


get_stale_folders () {
    servers=("$@")
    for server in "${servers[@]}";
    do
        echo "checking $server ..."
        for scratch in ${scratch_folders[@]}; do
            if [[ -d "/usr/$scratch/$server/" ]]; then
                df -h /usr/$scratch/$server/ | head -2 | tail -1
                if ((_CHECK_OPTION)); then
                    for dir in `find /usr/$scratch/$server/ -maxdepth 1 -mindepth 1 -type d -mtime +$DAYS`; do test `find $dir -type f -mtime -$DAYS -print -quit` || echo $dir && du -hs $dir; done
                fi
                if ((_DELETE_OPTION)); then
                    for dir in `find /usr/$scratch/$server/ -maxdepth 1 -mindepth 1 -type d -mtime +$DAYS`; do test `find $dir -type f -mtime -$DAYS -print -quit` || echo $dir && du -hs $dir && rm -rf $dir; done
                fi
            fi
        done
    done
}

_print_help() {
  cat <<HEREDOC
Usage: checkStorage.sh tortin | mont-fort | dahu | attelas | pisoc | arina | design | sassauna | vilan | dolent | gpu
Options:
  -c, --check                      Print list of old folders
  -d, --delete                     Delete old folders
HEREDOC
}

while [ ${#} -gt 0 ]
do
    __option="${1:-}"
    case "${__option}" in
        -d|--delete)
            _DELETE_OPTION=1;
        ;;
        -c|--check)
            _CHECK_OPTION=1;
        ;;
        tortin)
            SERVERS_OPTION=$TORTIN;
            _INFO_OPTION=1;
        ;;
        dahu)
            SERVERS_OPTION=$DAHU;
            _INFO_OPTION=1;
        ;;
        arina)
            SERVERS_OPTION=$ARINA;
            _INFO_OPTION=1;
        ;;
        attelas)
            SERVERS_OPTION=$ATTELAS;
            _INFO_OPTION=1;
        ;;
        pisoc)
            SERVERS_OPTION=$PISOC;
            _INFO_OPTION=1;
        ;;
        fenga)
            SERVERS_OPTION=$FENGA;
            _INFO_OPTION=1;
        ;;
        larain)
            SERVERS_OPTION=$LARAIN;
            _INFO_OPTION=1;
        ;;
        lagrev)
            SERVERS_OPTION=$LAGREV;
            _INFO_OPTION=1;
        ;;
        design)
            SERVERS_OPTION=$PISOC $FENGA $LARAIN $LAGREV;
            _INFO_OPTION=1;
        ;;
        sassauna)
            SERVERS_OPTION=$SASSAUNA;
            _INFO_OPTION=1;
        ;;
        vilan)
            SERVERS_OPTION=$VILAN;
            _INFO_OPTION=1;
        ;;
        dolent)
            SERVERS_OPTION=$DOLENT;
            _INFO_OPTION=1;
        ;;
        gpu)
            SERVERS_OPTION=$SASSAUNA $VILAN $OJOS;
            _INFO_OPTION=1;
        ;;
        mont-fort|montfort)
            SERVERS_OPTION=$MONTFORT;
            _INFO_OPTION=1;
        ;;
        *)
            _print_help; exit 1
        ;;
    esac
    shift
done


_main() {
    if ((_INFO_OPTION)); then
        get_stale_folders $SERVERS_OPTION
    else
        _print_help
    fi
}
_main "${@:-}"