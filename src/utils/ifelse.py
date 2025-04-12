"""If-else switch for hydra configs."""
import omegaconf

def ifelse(cond, vtrue, vfalse):
    if isinstance(cond, str):
        cstr = cond.capitalize()
        if cstr not in ["True", "False"]:
            raise ValueError("Received a str from config that should be a boolean. " \
            f"Only 'True','true','False','false' are accepted. Got {cond}")
        c = cstr == "True"
    elif isinstance(cond, bool): 
        c = cond 
    return vtrue if c else vfalse

omegaconf.OmegaConf.register_new_resolver("ifelse", ifelse)


# Tests

config = omegaconf.OmegaConf.create(
    {"cond":"true", 
     "result": "${ifelse:${cond},'this is true', 'this is false'}",
    }
)
assert config.result == "this is true"

config = omegaconf.OmegaConf.create(
    {"cond":"False", 
     "result": "${ifelse:${cond},'this is true', 'this is false'}",
    }
)
assert config.result == "this is false"

try: 
    config = omegaconf.OmegaConf.create(
    {"cond":"falsee", 
     "result": "${ifelse:${cond},'this is true', 'this is false'}",
    }
    )
    resolved_cfg = omegaconf.OmegaConf.to_container(config, resolve=True)
except ValueError as e: 
    error = e
# I wanted to check we have the right message but this is a bit troublesome, this will do.
assert error 

