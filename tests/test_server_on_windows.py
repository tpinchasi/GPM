"""The pool's secrets on Windows (D163): trusted by where they live, not by permission bits.

Windows has no permission bits Python can read — it reports every ordinary file as readable by
everyone — so the key files, the client CA's key and the provider credentials are taken from under
the user's profile only, as the host agent takes its settings (D154). Played here on any system,
with the profile a temporary folder.
"""

import os

import pytest
from gpm_server import private_files
from gpm_server.certs import CertRefused, ClientCA, make_ca
from gpm_server.credentials import CredentialStore, CredentialStoreUnsafe
from gpm_server.keys import KeyFileUnsafe, KeyStore


@pytest.fixture
def windows(tmp_path, monkeypatch):
    """A Windows machine whose user's profile is `tmp_path / "profile"`; `tmp_path / "C-root"`
    stands for a folder elsewhere, such as one made at the root of a drive."""
    profile = tmp_path / "profile"
    profile.mkdir()
    (tmp_path / "C-root").mkdir()
    monkeypatch.setattr(private_files, "WINDOWS", True)
    monkeypatch.setenv("USERPROFILE", str(profile))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    return profile


def everyones(path):
    """What Python on Windows reports for every ordinary file and folder."""
    path.chmod(0o777 if path.is_dir() else 0o666)
    return path


def test_a_key_file_under_the_profile_is_read_whatever_its_bits_say(windows):
    store = KeyStore(windows / "keys" / "app.keys")
    store.create("app")
    everyones(store.path)
    assert len(store.load()) == 1


def test_a_key_file_outside_the_profile_is_refused_saying_where_to_move_it(windows, tmp_path):
    store = KeyStore(tmp_path / "C-root" / "app.keys")
    store.create("app")
    with pytest.raises(KeyFileUnsafe, match="outside your profile") as refused:
        store.load()
    assert str(windows) in str(refused.value) and "chmod" not in str(refused.value)


def test_without_a_profile_to_go_by_a_key_file_is_refused(windows, monkeypatch):
    store = KeyStore(windows / "app.keys")
    store.create("app")
    monkeypatch.delenv("USERPROFILE")
    with pytest.raises(KeyFileUnsafe, match="cannot be checked"):
        store.load()


def test_a_link_out_of_the_profile_counts_as_where_it_leads(windows, tmp_path):
    real = KeyStore(tmp_path / "C-root" / "app.keys")
    real.create("app")
    os.symlink(real.path, windows / "app.keys")
    with pytest.raises(KeyFileUnsafe, match="outside your profile"):
        KeyStore(windows / "app.keys").load()


def test_the_client_cas_key_is_taken_from_the_profile_and_refused_elsewhere(windows, tmp_path):
    cert, key = make_ca(windows)
    everyones(key)
    assert ClientCA(cert, key).pem.startswith("-----BEGIN CERTIFICATE-----")
    elsewhere = tmp_path / "C-root" / "ca"
    elsewhere.mkdir()
    cert, key = make_ca(elsewhere)
    with pytest.raises(CertRefused, match="outside your profile"):
        ClientCA(cert, key)


def test_provider_credentials_under_the_profile_are_kept_and_read(windows):
    store = CredentialStore(windows / "provider-credentials")
    store.put("vast", "not-a-real-credential", {})
    everyones(store.directory)
    for entry in store.directory.iterdir():
        everyones(entry)
    everyones(windows)
    store.check()
    assert store.get("vast").value == "not-a-real-credential"


def test_provider_credentials_outside_the_profile_are_refused(windows, tmp_path):
    store = CredentialStore(tmp_path / "C-root" / "provider-credentials")
    with pytest.raises(CredentialStoreUnsafe, match="outside your profile"):
        store.put("vast", "not-a-real-credential", {})


def test_on_macos_and_linux_the_bits_still_decide(tmp_path, monkeypatch):
    monkeypatch.setattr(private_files, "WINDOWS", False)
    store = KeyStore(tmp_path / "app.keys")
    store.create("app")
    store.path.chmod(0o644)
    with pytest.raises(KeyFileUnsafe, match="readable by others"):
        store.load()
