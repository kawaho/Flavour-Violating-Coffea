while true; do
  read -p "Which setup? (local/lcg/conda): " setup
  
  case $setup in
    local)
      echo "-------------Setting up $setup virtual env-------------"
      source coffeaenv_local/bin/activate
      echo "-------------          Done            -------------"
      break
      ;;
    lcg)
      echo "-------------Setting up $setup virtual env-------------"
      source coffeaenv/bin/activate
      echo "-------------          Done            -------------"
      voms96
      break
      ;;
    conda)
      echo "-------------Setting up $setup virtual env-------------"
      conda activate coffeaenv_conda
      echo "-------------          Done            -------------"
      break
      ;;
    *)
      echo "-------------        Invalid setup        -------------"
      echo "-------------    Enter local/lcg/conda    -------------"
      ;;
  esac
done
