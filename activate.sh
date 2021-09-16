unset PYTHONPATH
voms-proxy-init --rfc --voms cms -valid 192:00
while true; do
  read -p "Which setup? (local/remote): " setup
  case $setup in
    local)
      echo "-------------Setting up $setup virtual env-------------"
      conda activate  my-coffea-env
      echo "-------------          Done            -------------"
      break
      ;;
    remote)
      echo "-------------Setting up $setup virtual env-------------"
      conda activate  remote-coffea-env
      echo "-------------          Done            -------------"
      break
      ;;
    *)
      echo "-------------        Invalid setup        -------------"
      echo "-------------     Enter local/remote      -------------"
      ;;
  esac
done
