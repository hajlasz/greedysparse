#!/usr/bin/env bash

set -euo pipefail

########################################
# Resolve base directory safely
########################################
__base_dir="${CONTAINER_IMAGES_FOLDER:-$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"

########################################
# Command + argument handling
########################################
__cmd="$(basename -- "${BASH_SOURCE[0]}")"

# Build safely quoted argument string
__args=""
for arg in "$@"; do
  printf -v qarg '%q' "${arg}"
  __args+=" ${qarg}"
done

RC_FILE=$(cat <<'EOF'
unset -f which
if [[ -f /ext3/env.sh ]]; then source /ext3/env.sh || exit 1; fi
export PYTHONNOUSERSITE=1
EOF
)

# If invoked via run-*, drop command name and default to shell
if [[ ${__cmd} = run-* ]]; then
  __cmd=""
  if [[ -z "${__args}" ]] || [[ "${__args}" == " /bin/bash" ]] || [[ "${__args}" == " bash" ]]; then
    __args="/bin/bash -rcfile <(echo \"${RC_FILE}\")";
    RC_FILE=""
  fi
fi
# __args="/bin/bash -rcfile <(echo 'unset -f which && source /ext3/env.sh || exit 1')";

########################################
# GPU detection
########################################
nv="" && if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

tmp_bind=""
if [[ "${SLURM_TMPDIR:-}" != "" ]] && [[ -d ${SLURM_TMPDIR} ]]; then
  tmp_bind="--bind ${SLURM_TMPDIR}:/tmp"
fi

########################################
# Execute container
########################################
exec singularity exec ${nv:-}  ${tmp_bind:-} \
--overlay ${__base_dir}/dice-20250525-ubuntu-24.04.2.sqf:ro \
${__base_dir}/ubuntu-24.04.2.sif \
/bin/bash -c "
${RC_FILE}
${__cmd} ${__args}
"
